import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--name", default=None, type=str, dest="name")
parser.add_argument("--template", default="MedT_Run1", type=str, dest="template")
parser.add_argument("--dataset", default="", type=str, dest="dataset")
args_parser = parser.parse_args()

name = args_parser.name
template_name = args_parser.template
dataset = args_parser.dataset

with open(f"./runs/{template_name}/run_train.tbi") as train_tbi_template:
    train_tbi_template = train_tbi_template.read()
    train_tbi_template = train_tbi_template.replace(template_name, name)

with open(f"./runs/{template_name}/run_train.sh") as train_sh_template:
    train_sh_template = train_sh_template.read()
    train_sh_template = train_sh_template.replace(template_name, name)

with open(f"./runs/{template_name}/run_test.tbi") as test_tbi_template:
    test_tbi_template = test_tbi_template.read()
    test_tbi_template = test_tbi_template.replace(template_name, name)

with open(f"./runs/{template_name}/run_test.sh") as test_sh_template:
    test_sh_template = test_sh_template.read()
    test_sh_template = test_sh_template.replace(template_name, name)

os.makedirs(f"./runs/{name}", exist_ok=True)
os.makedirs(f"./runs/{name}/tensorboard", exist_ok=True)

with open(f"./runs/{name}/run_train.tbi", "w") as train_tbi:
    train_tbi.write(train_tbi_template)

with open(f"./runs/{name}/run_train.sh", "w") as train_sh:
    train_sh.write(train_sh_template)

with open(f"./runs/{name}/run_test.tbi", "w") as test_tbi:
    test_tbi.write(test_tbi_template)

with open(f"./runs/{name}/run_test.sh", "w") as test_sh:
    test_sh.write(test_sh_template)
