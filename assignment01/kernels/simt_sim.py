"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""


def run(program):
    regs = list(range(32))

    def exec_prog(prog, mask):
        cycles = 0
        for inst in prog:
            op = inst[0]

            if op == "add":
                k = inst[1]
                for i, active in enumerate(mask):
                    if active:
                        regs[i] += k
                cycles += 1

            elif op == "mul":
                k = inst[1]
                for i, active in enumerate(mask):
                    if active:
                        regs[i] *= k
                cycles += 1

            elif op == "if_lt":
                _, t, then_prog, else_prog = inst
                then_mask = [active and regs[i] < t for i, active in enumerate(mask)]
                else_mask = [active and regs[i] >= t for i, active in enumerate(mask)]

                if any(then_mask):
                    cycles += exec_prog(then_prog, then_mask)
                if any(else_mask):
                    cycles += exec_prog(else_prog, else_mask)

            else:
                raise ValueError(f"unknown instruction: {op}")

        return cycles

    cycles = exec_prog(program, [True] * 32)
    return regs, cycles
