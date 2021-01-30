import numpy as np

rule_number = int(input("Enter the rule: "))

rng = np.random.RandomState(42)
initial = rng.randint(0, 2, 20)  # half open [low: 0, high: 1)
# initial = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print("Random array:", initial)


def rule_index(triplet):
    L, C, R = triplet
    index = 7 - (4 * L + 2 * C + R)
    return int(index)


def CA_Run(initial_state, n_steps, rule_by_user):
    rule_string = np.binary_repr(rule_by_user, 8)
    rule = np.array([int(bit) for bit in rule_string])
    print("Rule of", rule_number, rule)

    m_cells = len(initial_state)
    CA_run = np.zeros((n_steps, m_cells))
    CA_run[0, :] = initial_state

    for step in range(1, n_steps):
        all_triplets = np.stack(
            [
                np.roll(CA_run[step - 1, :], 1),
                CA_run[step - 1, :],
                np.roll(CA_run[step - 1, :], -1),
            ]
        )
        CA_run[step, :] = rule[np.apply_along_axis(rule_index, 0, all_triplets)]

    return CA_run


data = CA_Run(initial, 20, rule_number)
print(data)
