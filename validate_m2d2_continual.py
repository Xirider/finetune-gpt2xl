from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import json
import torch
import argparse


def main(args):
    task_idx = int(args.model_path.split("task")[1])
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to("cuda")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_path)
    output_file = f"{args.model_path}_results.json"
    list_of_datasets = ["Art", "Culture_and_the_arts", "Culture_and_the_arts__Culture_and_Humanities", "Culture_and_the_arts__Games_and_Toys", "Culture_and_the_arts__Mass_media", "Culture_and_the_arts__Performing_arts", "Culture_and_the_arts__Sports_and_Recreation", "Culture_and_the_arts__The_arts_and_Entertainment", "Culture_and_the_arts__Visual_arts", "General_referece", "General_referece__Further_research_tools_and_topics", "General_referece__Reference_works", "Health_and_fitness", "Health_and_fitness__Exercise", "Health_and_fitness__Health_science", "Health_and_fitness__Human_medicine", "Health_and_fitness__Nutrition", "Health_and_fitness__Public_health", "Health_and_fitness__Self_care", "History_and_events", "History_and_events__By_continent", "History_and_events__By_period", "History_and_events__By_region", "Human_activites", "Human_activites__Human_activities", "Human_activites__Impact_of_human_activity", "Mathematics_and_logic", "Mathematics_and_logic__Fields_of_mathematics", "Mathematics_and_logic__Logic", "Mathematics_and_logic__Mathematics", "Natural_and_physical_sciences", "Natural_and_physical_sciences__Biology", "Natural_and_physical_sciences__Earth_sciences", "Natural_and_physical_sciences__Nature", "Natural_and_physical_sciences__Physical_sciences", "Philosophy", "Philosophy_and_thinking", "Philosophy_and_thinking__Philosophy", "Philosophy_and_thinking__Thinking", "Religion_and_belief_systems", "Religion_and_belief_systems__Allah", "Religion_and_belief_systems__Belief_systems", "Religion_and_belief_systems__Major_beliefs_of_the_world", "Society_and_social_sciences", "Society_and_social_sciences__Social_sciences", "Society_and_social_sciences__Society", "Technology_and_applied_sciences", "Technology_and_applied_sciences__Agriculture", "Technology_and_applied_sciences__Computing", "Technology_and_applied_sciences__Engineering", "Technology_and_applied_sciences__Transport", "alg-geom", "ao-sci", "astro-ph", "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE", "astro-ph.IM", "astro-ph.SR", "astro-ph_l1", "atom-ph", "bayes-an", "chao-dyn", "chem-ph", "cmp-lg", "comp-gas", "cond-mat", "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci", "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft", "cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con", "cond-mat_l1", "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY", "cs_l1", "dg-ga", "econ.EM", "econ.GN", "econ.TH", "econ_l1", "eess.AS", "eess.IV", "eess.SP", "eess.SY", "eess_l1", "eval_sets", "funct-an", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GR", "math.GT", "math.HO", "math.IT", "math.KT", "math.LO", "math.MG", "math.MP", "math.NA", "math.NT", "math.OA", "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG", "math.SP", "math.ST", "math_l1", "mtrl-th", "nlin.AO", "nlin.CD", "nlin.CG", "nlin.PS", "nlin.SI", "nlin_l1", "nucl-ex", "nucl-th", "only_text", "only_txt2", "patt-sol", "physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus", "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph", "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det", "physics.med-ph", "physics.optics", "physics.plasm-ph", "physics.pop-ph", "physics.soc-ph", "physics.space-ph", "physics_l1", "plasm-ph", "q-alg", "q-bio", "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO", "q-bio_l1", "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR", "q-fin_l1", "quant-ph", "solv-int", "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH", "stat_l1", "supr-con"]

    try:
        with open(output_file, "r") as f:
            output_ppls = json.load(f)
    except:
        output_ppls = {}

    for ds in tqdm(list_of_datasets[:task_idx+1]):
        if ds in output_ppls:
            continue
        dataset = load_dataset("machelreid/m2d2", ds,
            # cache_dir=os.path.join(os.environ["HOME"], "storage4", ".cache"),
        )
        encodings = tokenizer("\n".join(dataset["train"]["text"]), return_tensors="pt")
        with torch.no_grad():
            ppl = eval_ppl(model, encodings, stride=1024)
        output_ppls[ds] = ppl.item()
        print(output_ppls)
        with open(output_file, "w") as f:
            json.dump(output_ppls, f)


def eval_ppl(model, encodings, stride, device="cuda"):
    max_length = model.config.n_positions
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    main(args)
