import argparse
import mteb

parser = argparse.ArgumentParser("MTEB")
parser.add_argument("--task", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--revision", type=str, default="")
parser.add_argument("--model_root", default=None, type=str)
args = parser.parse_args()
print(args)

# tasks = mteb.get_benchmark("MTEB(eng, classic)")

if args.model_root is not None:
    model_name = args.model_root.split('/')[-1]
    model = mteb.get_model('dinov2_vlm/' + model_name, revision="", model_root=args.model_root) # load model using registry implementation if available, otherwise use SentenceTransformers

else:
    model = mteb.get_model(args.model_name, revision=args.revision) # load model using registry implementation if available, otherwise use SentenceTransformers

tasks = mteb.get_tasks(tasks = [args.task], languages = ["eng"])
# tasks = mteb.get_benchmark("MTEB(eng, classic)")

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, verbosity=3)
