import argparse

from src.qcbm_train import build_ansatz, QCBMConfig


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Print QCBM circuit.")
    parser.add_argument("--qubits", type=int, default=16, help="Number of qubits.")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers.")
    parser.add_argument("--output", default="qcbm_circuit.png", help="Output image path.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    n_params = args.qubits * args.layers
    theta = [0.0] * n_params
    circuit = build_ansatz(args.qubits, args.layers, theta)
    try:
        fig = circuit.draw(output="mpl")
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved circuit image to {args.output}")
    except Exception as exc:
        print("Could not render circuit image. Printing text circuit instead.")
        print(circuit.draw())
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
