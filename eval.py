from sentence_transformers import SentenceTransformer
import mteb
import argparse


def main(args):
    model_name = args.model_name
    model = SentenceTransformer(model_name)
    evaluation = mteb.MTEB(tasks=["AmazonCounterfactualClassification",
                                "AmazonReviewsClassification",
                                "LivedoorNewsClustering.v2",
                                "MewsC16JaClustering",
                                "MIRACLReranking",
                                "NLPJournalAbsIntroRetrieval",
                                "NLPJournalTitleAbsRetrieval",
                                "NLPJournalTitleIntroRetrieval",
                                "JSICK",
                                "JSTS"])
    evaluation.run(model, output_folder="output/mteb_results",
                            batch_size=args.batch_size, num_workers=args.num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTEB evaluation on a model.")
    parser.add_argument("--model_name", type=str, default="cl-nagoya/ruri-v3-pt-30m",help="Name of the model to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8,help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4,help="Number of workers for data loading.")
    args = parser.parse_args()
    main(args)