if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from script.evaluate_inv_parallel import evaluate, evaluate_fast
    from config.locomotion_config import Config
    from params_proto.hyper import Sweep

    sweep = Sweep(RUN, Config).load("default_inv.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        # thunk = instr(evaluate, **kwargs)
        thunk = instr(evaluate_fast, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()