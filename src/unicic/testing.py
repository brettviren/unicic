def pairwise_check_arrays(data, epsilon=1e-6):
    for nn, bytoy in data.items():
        # print(f'\nNN:{nn}')
        toys = list(bytoy)
        if len(toys) < 2:
            continue
        for t1, t2 in zip(toys[:-1], toys[1:]):
            n1 = bytoy[t1]
            n2 = bytoy[t2]
            # print(f'{t1}: {n1}')
            # print(f'{t2}: {n2}')
            for v1,v2 in zip(n1.reshape(-1),n2.reshape(-1)):
                dv = abs(v1 - v2)/(abs(v1)+abs(v2))
                if dv < epsilon:
                    continue
                print(f'\nindex:{nn}')
                print(f'\n{t1.__name__}:{v1}')
                print(f'\n{t2.__name__}:{v2}')
                print(f'\ndiff:{dv}')
                raise ValueError(f'large difference {dv} between {t1.__name__} and {t2.__name__} for {nn}: {v1} != {v2}')
    
