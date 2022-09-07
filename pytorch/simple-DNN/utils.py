import argparse

def _setup_parser():
    parser = argparse.ArgumentParser(description="Aprendendo a fazer um autoencoder simples")

    # Train the simple autoencoder
    parser.add_argument('--train', action='store_true', help="Run the full pipeline and train the simple autoencoder")

    return parser.parse_args()

