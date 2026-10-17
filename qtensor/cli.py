"""
QTensor CLI
"""
import argparse
from qtensor.model import QTensorModel

def main():
    parser = argparse.ArgumentParser(description="QTensor Client SDK CLI")
    parser.add_argument("model", help="Model ID or directory path")
    parser.add_argument("--prompt", default="def test():", help="Inference prompt")
    args = parser.parse_args()

    model = QTensorModel.from_pretrained(args.model)
    output = model.generate(args.prompt)
    print(output)

if __name__ == "__main__":
    main()
