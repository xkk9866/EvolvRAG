"""Paired checks that the manuscript needs and that the main grid did not print."""

from run_experiment import (
    DEFAULT_POBS,
    DEFAULT_RATE,
    SEEDS_TEST,
    generate_trajectory,
    mean_std,
    pack_metrics,
    run_method,
    signflip_p,
)


def sp_of(method, worlds, budget=None):
    scores = []
    for seed, world in zip(SEEDS_TEST, worlds):
        counts, _ = run_method(world, method, seed, budget=budget)
        scores.append(pack_metrics(counts)["sp"])
    return scores


def main():
    worlds = {
        p: [generate_trajectory(seed, DEFAULT_RATE, p) for seed in SEEDS_TEST]
        for p in (1.0, DEFAULT_POBS)
    }
    full_c = sp_of("Central", worlds[1.0])
    full_m = sp_of("Majority", worlds[1.0])
    diffs = [a - b for a, b in zip(full_c, full_m)]
    mu, sd = mean_std(diffs)
    print("full obs Central-Majority SP", f"{mu:+.4f}", f"sd {sd:.4f}", f"p {signflip_p(diffs):.4f}")
    bud = sp_of("Central", worlds[DEFAULT_POBS], budget=8)
    maj = sp_of("Majority", worlds[DEFAULT_POBS])
    diffs = [a - b for a, b in zip(bud, maj)]
    mu, sd = mean_std(diffs)
    print("budget8 Central-Majority SP", f"{mu:+.4f}", f"sd {sd:.4f}", f"p {signflip_p(diffs):.4f}")
    # Logged key on the first test seed, same cell the trace used.
    world = worlds[DEFAULT_POBS][0]
    _counts, answers = run_method(world, "Central", SEEDS_TEST[0])
    slot = 55
    child = 135
    print("seed", SEEDS_TEST[0], "epoch3 central parent", answers[2][0][slot])
    print("epoch3 central child", answers[2][0][child])
    print("all agents parent", [answers[2][a][slot] for a in range(8)])


if __name__ == "__main__":
    main()
