import consensus as CS4A
import argparse
def run_consensus_iteration(reads, initial_reference, num_iterations, output_folder):
    current_reference = initial_reference
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        output_file = CS4A.get_consensus_sequence(reads, current_reference, f"{output_folder}/iteration_{i+1}")
        current_reference = output_file
    return current_reference

def main():
    parser = argparse.ArgumentParser(description="Generate consensus sequence from reads")
    parser.add_argument("-b", required=True, help="Path to the initial reference file")
    parser.add_argument("-r", required=True, help="Path to the input reads file")
    parser.add_argument("-outdir", required=True, help="Path to the output folder")
    parser.add_argument("-n", type=int, default=3, help="Number of iterations to perform")
    args = parser.parse_args()

    final_consensus = run_consensus_iteration(args.r, args.b, args.n, args.outdir)
    print(f"Final consensus sequence saved at: {final_consensus}")

if __name__ == "__main__":
    main()