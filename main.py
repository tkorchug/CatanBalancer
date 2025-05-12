import argparse

from catan_board_balancer import generate_board


def main():
    parser = argparse.ArgumentParser(description="Catan Board Balancer")
    parser.add_argument(
        "--version",
        choices=["original", "got"],
        help="Choose the version of the Catan board: 'original' or 'got'"
    )
    args = parser.parse_args()

    # Call the board balancer with the selected version
    generate_board(args.version)


if __name__ == "__main__":
    main()
