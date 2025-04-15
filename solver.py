import networkx as nx
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpInteger, LpBinary, PULP_CBC_CMD, LpStatus, value
from pprint import pprint


# Нахождение критичных цепочек процессов и вычисление минимально возможного времени выполнения всех процессов
def find_all_critical_paths(processes: dict):
    G = nx.DiGraph()
    procs = sorted(processes.keys())
    G.add_nodes_from(procs)
    for i in procs:
        dependencies = processes[i][2]
        if dependencies:
            G.add_edges_from([(dep, i) for dep in dependencies])
    sorted_processes = nx.topological_sort(G)
    for process in processes:
        processes[process][3] = 0
        processes[process][4] = 0
    paths = {process: [] for process in processes}
    # Расчёт времени выполнения для каждого процесса
    for process in sorted_processes:
        duration = processes[process][1]
        max_preceding_time = max((processes[pred][4] for pred in
                                  processes[process][2]), default=0)
        processes[process][3] = max_preceding_time
        processes[process][4] = duration + max_preceding_time
        # Обновление путей
        for pred in processes[process][2]:
            if processes[pred][4] + duration == processes[process][4]:
                paths[process].append(pred)
    # Определение всех максимальных путей
    max_time = max(processes[p][4] for p in processes)
    critical_paths = []
    for process, time in processes.items():
        if processes[process][4] == max_time:
            critical_paths.extend(find_paths(process, paths))
    return critical_paths, max_time


# Вспомогательная функция которая нужна для работы предыдущей
def find_paths(process, paths):
    if not paths[process]:
        return [[process]]
    subpaths = []
    for pred in paths[process]:
        subpaths.extend(find_paths(pred, paths))
    return [[process] + path for path in subpaths]


# Нахождение максимально возможных задержек для процессов без нарушения лимита времени
def find_max_lags(processes: dict, forbidden_processes: set, time_limit):
    G = nx.DiGraph()
    procs = sorted(processes.keys())
    G.add_nodes_from(procs)
    for i in procs:
        dependencies = processes[i][2]
        if dependencies:
            G.add_edges_from([(dep, i) for dep in dependencies])
    sorted_processes = list(nx.topological_sort(G))
    lags = dict()
    last_times = dict()
    for p in sorted_processes[::-1]:
        next_processes = list(G.successors(p))
        if next_processes:
            last_time = min(last_times[p] - processes[p][1] for p in next_processes)
        else:
            last_time = time_limit
        last_times[p] = last_time
    for p in sorted_processes:
        if p not in forbidden_processes:
            prev_proc = list(G.predecessors(p))
            lag = last_times[p] - processes[p][1]
            if prev_proc:
                lag -= max(last_times[p] for p in prev_proc)
            if lag > 0:
                lags[p] = lag
    return lags


# заполнение словаря процессов по данным из файла
# Словарь процессов имеет структуру
# key id: value [0-id 1-duration 2-dependencies 3-begin 4-end]
def parse_file(filename):
    processes = dict()
    # processes[0] = [0, 0, tuple(), 0, 0]
    N = 0
    with open(filename) as f:
        for row in f:
            row_id, row_duration, row_dependencies = row.strip().split('\t')
            row_id = int(row_id)
            row_duration = int(row_duration)
            row_dependencies = tuple(map(int, row_dependencies.split(';')))
            if row_dependencies != (0,):
                processes[row_id] = [row_id, row_duration, row_dependencies, 0, 0]
            else:
                processes[row_id] = [row_id, row_duration, tuple(), 0, 0]
            N += 1
    return N, processes

def parse_file_6(filename):
    processes = dict()
    with open(filename) as f:
        for row in f:
            if 't' not in row:
                row_id, row_duration, row_dependencies = row.strip().split('\t')
                row_id = int(row_id)
                row_duration = int(row_duration)
                row_dependencies = tuple(map(int, row_dependencies.split(';')))
                if row_dependencies != (0,):
                    processes[row_id] = [row_id, row_duration, row_dependencies, 0, 0]
                else:
                    processes[row_id] = [row_id, row_duration, tuple(), 0, 0]
            else:
                row_id, row_duration, row_dependencies = row.strip().split('\t')
                row_id = int(row_id)
                # row_duration = int(row_duration)
                row_dependencies = tuple(map(int, row_dependencies.split(';')))
                if row_dependencies != (0,):
                    t_row = [row_id, 0, row_dependencies, 0, 0]
                else:
                    t_row = [row_id, 0, tuple(), 0, 0]
    return processes, t_row


# Нахождение максимально возможного числа процессов, которые могут выполняться одновременно
def maximize_peak_parallelism(processes: dict, time_limit: int) -> int:
    prob = LpProblem("MaximizePeakParallelism", LpMaximize)

    procs = sorted(processes.keys())
    durations = {i: processes[i][1] for i in procs}
    dependencies = {i: processes[i][2] for i in procs}

    # Переменные начала процесса
    s = {i: LpVariable(f"s_{i}", 0, time_limit - durations[i], cat=LpInteger) for i in procs}

    # Переменные активности процесса в момент времени t
    a = {
        (i, t): LpVariable(f"a_{i}_{t}", cat=LpBinary)
        for i in procs for t in range(time_limit)
    }

    # Булевы переменные: достигается ли пик в момент t
    is_peak_time = {
        t: LpVariable(f"is_peak_time_{t}", cat=LpBinary)
        for t in range(time_limit)
    }

    # Переменная пика
    peak = LpVariable("peak", 0, time_limit, cat=LpInteger)

    # Ограничения по зависимостям: s[i] >= s[j] + duration[j]
    for i in procs:
        for j in dependencies[i]:
            prob += s[i] >= s[j] + durations[j], f"dep_{j}_to_{i}"

    # Связываем a[i, t] с временем начала s[i] и длительностью
    for i in procs:
        d_i = durations[i]
        for t in range(time_limit):
            # a[i, t] == 1 если s[i] <= t < s[i] + d_i
            # Заменим строгое неравенство на эквивалентное >=
            prob += s[i] <= t + (1 - a[i, t]) * time_limit, f"active_start_{i}_{t}"
            prob += s[i] + d_i >= t + 1 - (1 - a[i, t]) * time_limit, f"active_end_{i}_{t}"

    # Ограничения на пик: в момент t сумма активных процессов >= peak * is_peak_time[t]
    big_M = len(procs)
    for t in range(time_limit):
        prob += lpSum(a[i, t] for i in procs) >= peak - (1 - is_peak_time[t]) * big_M, f"peak_def_{t}"
    prob += lpSum(is_peak_time[t] for t in range(time_limit)) == 1, "one_peak_time"

    # Требуем, чтобы пик был достигнут хотя бы один раз
    prob += lpSum(is_peak_time[t] for t in range(time_limit)) >= 1, "at_least_one_peak"

    # Целевая функция — максимизировать peak
    prob += peak, "maximize_peak"

    # Решение
    prob.solve(PULP_CBC_CMD(msg=0))

    # Вернём значение peak
    return int(value(peak))


def maximize_peak_duration(processes, time_limit, peak):
    prob = LpProblem("MaximizePeakDuration", LpMaximize)

    procs = list(processes.keys())

    # Время начала каждого процесса
    s = {
        i: LpVariable(f"s_{i}", lowBound=0, upBound=time_limit - processes[i][1], cat=LpInteger)
        for i in procs
    }

    # Бинарные переменные активности процесса i в момент t
    a = {
        (i, t): LpVariable(f"a_{i}_{t}", cat=LpBinary)
        for i in procs for t in range(time_limit)
    }

    # Переменные начала и конца пикового интервала
    peak_start = LpVariable("peak_start", lowBound=0, upBound=time_limit - 1, cat=LpInteger)
    peak_end = LpVariable("peak_end", lowBound=1, upBound=time_limit, cat=LpInteger)
    peak_duration = LpVariable("peak_duration", lowBound=0, upBound=time_limit, cat=LpInteger)

    # Целевая функция — максимизировать длительность пика
    prob += peak_duration, "Maximize_peak_duration"

    # Связь начала и конца пика
    prob += peak_end == peak_start + peak_duration, "peak_duration_def"

    # Ограничения на активность процессов
    for i in procs:
        d_i = processes[i][1]
        for t in range(time_limit):
            prob += s[i] <= t + (1 - a[i, t]) * time_limit, f"active_start_{i}_{t}"
            prob += s[i] + d_i >= t + 1 - (1 - a[i, t]) * time_limit, f"active_end_{i}_{t}"

    # Переменные is_in_peak[t] и уточнённые условия включения t в интервал пика
    is_in_peak = {
        t: LpVariable(f"is_in_peak_{t}", cat=LpBinary)
        for t in range(time_limit)
    }
    M = time_limit  # Большое число

    # Ограничения на процессы, выполняющиеся в пиковом интервале
    for t in range(time_limit):
        # t ∈ [peak_start, peak_end) ⇔ is_in_peak[t] = 1
        prob += peak_start <= t + (1 - is_in_peak[t]) * M, f"peak_start_condition_{t}"
        prob += t <= peak_end - 1 + (1 - is_in_peak[t]) * M, f"peak_end_condition_{t}"

        # В пиковом интервале должно выполняться ровно peak процессов
        prob += lpSum(a[i, t] for i in procs) == peak * is_in_peak[t], f"peak_constraint_{t}"

    # Убедимся, что переменные is_in_peak точно соответствуют заданному пику
    prob += lpSum(is_in_peak[t] for t in range(time_limit)) == peak_duration, "correct_peak_count"

    # Ограничения на зависимости между процессами
    for i in procs:
        for dep in processes[i][2]:
            prob += s[i] >= s[dep] + processes[dep][1], f"dependency_{dep}_to_{i}"

    # Решение
    prob.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob.status] != "Optimal":
        print("Не удалось найти оптимальное решение.")
        return None, None

    start_times = {i: int(value(s[i])) for i in procs}
    max_peak_duration = int(value(peak_duration))

    return start_times, max_peak_duration


# Нахождение максимального количества процессов, которые завершатся за определенное время
def maximum_on_time(processes, time_limit):
    G = nx.DiGraph()
    procs = sorted(processes.keys())
    G.add_nodes_from(procs)
    for i in procs:
        dependencies = processes[i][2]
        if dependencies:
            G.add_edges_from([(dep, i) for dep in dependencies])
    sorted_processes = nx.topological_sort(G)
    for process in processes:
        processes[process][3] = 0
        processes[process][4] = 0
    # Расчёт времени выполнения для каждого процесса
    for process in sorted_processes:
        duration = processes[process][1]
        max_preceding_time = max((processes[pred][4] for pred in
                                  processes[process][2]), default=0)
        processes[process][3] = max_preceding_time
        processes[process][4] = duration + max_preceding_time
    # Подсчет количества процессов, которые завершатся за указанное время
    amount = 0
    for proc in processes:
        if processes[proc][4] <= time_limit:
            amount += 1
    return amount

def minimal_with_lag_for_dependent(processes: dict, lag:int):
    G = nx.DiGraph()
    procs = sorted(processes.keys())
    G.add_nodes_from(procs)
    for i in procs:
        dependencies = processes[i][2]
        if dependencies:
            G.add_edges_from([(dep, i) for dep in dependencies])
    sorted_processes = nx.topological_sort(G)
    for process in processes:
        processes[process][3] = 0
        processes[process][4] = 0
    # Расчёт времени выполнения для каждого процесса
    for process in sorted_processes:
        duration = processes[process][1]
        max_preceding_time = max((processes[pred][4] for pred in
                                  processes[process][2]), default=0)
        processes[process][3] = max_preceding_time + lag * (max_preceding_time != 0)
        processes[process][4] = duration + processes[process][3]
    return max(processes[pr][4] for pr in processes)

def maximal_t_for_N_processes_on_T(processes: dict, N, T, t_row):
    t_id = t_row[0]
    processes[t_id] = t_row
    t = 1
    processes[t_id][1] = t
    current_N = maximum_on_time(processes, T)
    while current_N >= N:
        t += 1
        processes[t_id][1] = t
        current_N = maximum_on_time(processes, T)
    return t - 1

def minimal_t_for_all_on_T(processes, T, t_row):
    t_id = t_row[0]
    processes[t_id] = t_row
    t = 1
    processes[t_id][1] = t
    _, time_limit = find_all_critical_paths(processes)
    while time_limit <= T:
        t += 1
        processes[t_id][1] = t
        _, time_limit = find_all_critical_paths(processes)
    return t - 1

def amount_with_oddity(processes: dict, oddity):
    G = nx.DiGraph()
    procs = sorted(processes.keys())
    G.add_nodes_from(procs)
    for i in procs:
        dependencies = processes[i][2]
        if dependencies:
            G.add_edges_from([(dep, i) for dep in dependencies])
    sorted_processes = nx.topological_sort(G)
    for process in processes:
        processes[process][3] = 0
        processes[process][4] = 0
    paths = {process: [] for process in processes}
    # Расчёт времени выполнения для каждого процесса
    for process in sorted_processes:
        duration = processes[process][1]
        max_preceding_time = max((processes[pred][4] for pred in
                                  processes[process][2]), default=0)
        processes[process][3] = max_preceding_time
        processes[process][4] = duration + max_preceding_time
        # Обновление путей
        for pred in processes[process][2]:
            if processes[pred][4] + duration == processes[process][4]:
                paths[process].append(pred)
    amount = len([proc for proc in processes if processes[proc][4] % 2 == oddity])
    return amount

def solver():
    file = input('Введите имя файла(.txt):')
    print('Выберите тип задачи:')
    print('1) Найти максимальную продолжительность времени, в течение которого возможно одновременное \n'
          '   выполнение максимального количества процессов, при условии, что все независимые друг от \n'
          '   друга процессы могут выполняться параллельно, а время завершения каждого процесса минимально.')
    print('2) Определить минимальное время, через которое завершится выполнение всей совокупности процессов, \n'
          '   при условии, что все независимые друг от друга процессы могут выполняться параллельно.')
    print('3) Определить максимальную продолжительность отрезка времени (в мс), в течение которого \n'
          '   возможно одновременное выполнение N процессов, при условии, что все независимые друг \n'
          '   от друга процессы могут выполняться параллельно.')
    print('4) Определите максимальное количество процессов, которые могут завершиться по прошествии T мс')
    print('5) Определить минимальное время, через которое завершится выполнение всей совокупности процессов, \n'
          '   при условии, что все независимые друг от друга процессы могут выполняться параллельно.\n'
          '   Зависимый процесс начинается с задержкой T')
    print('6) Определите максимально возможное целочисленное t (время выполнения процесса), \n'
          '   при котором выполнение первых N процессов (при условии, что все независимые друг \n'
          '   от друга процессы могут выполняться параллельно и один процесс может сменять \n'
          '   другой, завершившийся мгновенно) завершилось не более чем за T мс.')
    print('7) Определите количество процессов минимальное время выполнения каждого из которых\n'
          '   является чётным / нечётным числом.')
    print('8) Определите минимально возможное целочисленное t (время выполнения процесса), \n'
          '   при котором выполнение всех процессов завершилось не ранее чем за T мс.')
    print('9) Определите максимально возможное целочисленное t (время выполнения процесса), \n'
          '   при котором выполнение всех процессов завершилось за T мс.')
    print('10) Найти максимальную продолжительность времени, в течение которого возможно одновременное \n'
          '   выполнение максимального количества процессов, при условии, что все независимые друг от \n'
          '   друга процессы могут выполняться параллельно(без ограничения минимального времени)')

    type_of_task = int(input('Введите число 1..10:'))
    if type_of_task not in [6, 8, 9]:
        N, processes = parse_file(file)
        critical_paths, time_limit = find_all_critical_paths(processes)
    if type_of_task == 1:
        peak = maximize_peak_parallelism(processes, time_limit)
        start_times, duration = maximize_peak_duration(processes, time_limit, peak)
        print("Максимальная продолжительность пика:", duration)
    elif type_of_task == 2:
        print('Минимальное время завершения всех процессов:', time_limit)
    elif type_of_task == 3:
        peak = int(input('Введите количество процессов:'))
        coeff = float(input('Введите коэффициент для максимального времени\n'
                            '(лучше начать от 1.0 и постепенно увеличивать до 2.0):'))
        start_times, duration = maximize_peak_duration(processes, int(time_limit * coeff), peak)
        print(f"Максимальная продолжительность {peak} процессов:", duration)
        print('Время начала процессов:')
        pprint(start_times)
    elif type_of_task == 4:
        time_limit = int(input('Введите время за которое должны завершиться процессы:'))
        amount = maximum_on_time(processes, time_limit)
        print(f"Максимальное количество процессов за {time_limit}:", amount)
    elif type_of_task == 5:
        lag = int(input('Введите длительность задержки перед зависимым процессом:'))
        print(f'Минимальное время завершения всех процессов с '
              f'задержкой {lag}: {minimal_with_lag_for_dependent(processes, lag)}')
    elif type_of_task == 6:
        # https://education.yandex.ru/ege/task/bc1a1196-41b4-47d0-9502-d9b70a7f227c
        processes, t_row = parse_file_6(file)
        N = int(input('Введите число процессов, которое должно быть выполнено:'))
        T = int(input('Введите время за которое, эти процессы должны быть выполнены:'))
        print(f'Максимальное время t:{maximal_t_for_N_processes_on_T(processes, N, T, t_row)}')
    elif type_of_task == 7:
        oddity = int(input('Введите чётность(0 - чётное, 1 - нечётное):'))
        amount = amount_with_oddity(processes, oddity)
        print(f'Количество процессов с четностью {oddity}: {amount}')
    elif type_of_task == 8:
        # https://education.yandex.ru/ege/task/b67b9e16-3668-4cc7-bc88-f83a27bc7031
        processes, t_row = parse_file_6(file)
        T = int(input('Введите время за которое, эти процессы должны быть выполнены:'))
        print(f'Минимальное время t:{minimal_t_for_all_on_T(processes, T, t_row)}')
    elif type_of_task == 9:
        # https://education.yandex.ru/ege/task/5a8c943b-648e-4307-b8d5-a3586f660605
        processes, t_row = parse_file_6(file)
        N = len(processes.keys()) + 1
        T = int(input('Введите время за которое, эти процессы должны быть выполнены:'))
        print(f'Максимальное время t:{maximal_t_for_N_processes_on_T(processes, N, T, t_row)}')
    elif type_of_task == 10:
        # Иглин 4 пробник, файл 22_4.txt Максимальное время 9 для 6 процессов
        coeff = float(input('Введите коэффициент для максимального времени\n'
                            '(лучше начать от 1.0 и постепенно увеличивать до 2.0):'))
        peak = maximize_peak_parallelism(processes, int(time_limit * coeff))
        start_times, duration = maximize_peak_duration(processes, int(time_limit * coeff), peak)
        print(f"Максимальная продолжительность {peak} процессов:", duration)
        print('Время начала процессов:')
        pprint(start_times)



solver()

# Не решаются задачи Шастина и PRO10EGE с большими значениями для временных
# отрезков их можно попробовать решить уменьшив длительность в 100-1000 раз
# и посмотреть, как будут располагаться процессы, а затем перейти к исходным.
# https://education.yandex.ru/ege/task/d4658b6e-671b-4165-8c34-d4607b247c2b
# https://education.yandex.ru/ege/task/1b0cd2c9-047a-4644-91ce-8e1f38b23da9
# https://education.yandex.ru/ege/task/250cf1cc-5326-4025-b07c-9f4862d5904e
