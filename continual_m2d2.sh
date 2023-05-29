#!/bin/bash

# Define the list of dataset names
datasets=("Art" "Culture_and_the_arts" "Culture_and_the_arts__Culture_and_Humanities" "Culture_and_the_arts__Games_and_Toys" "Culture_and_the_arts__Mass_media" "Culture_and_the_arts__Performing_arts" "Culture_and_the_arts__Sports_and_Recreation" "Culture_and_the_arts__The_arts_and_Entertainment" "Culture_and_the_arts__Visual_arts" "General_referece" "General_referece__Further_research_tools_and_topics" "General_referece__Reference_works" "Health_and_fitness" "Health_and_fitness__Exercise" "Health_and_fitness__Health_science" "Health_and_fitness__Human_medicine" "Health_and_fitness__Nutrition" "Health_and_fitness__Public_health" "Health_and_fitness__Self_care" "History_and_events" "History_and_events__By_continent" "History_and_events__By_period" "History_and_events__By_region" "Human_activites" "Human_activites__Human_activities" "Human_activites__Impact_of_human_activity" "Mathematics_and_logic" "Mathematics_and_logic__Fields_of_mathematics" "Mathematics_and_logic__Logic" "Mathematics_and_logic__Mathematics" "Natural_and_physical_sciences" "Natural_and_physical_sciences__Biology" "Natural_and_physical_sciences__Earth_sciences" "Natural_and_physical_sciences__Nature" "Natural_and_physical_sciences__Physical_sciences" "Philosophy" "Philosophy_and_thinking" "Philosophy_and_thinking__Philosophy" "Philosophy_and_thinking__Thinking" "Religion_and_belief_systems" "Religion_and_belief_systems__Allah" "Religion_and_belief_systems__Belief_systems" "Religion_and_belief_systems__Major_beliefs_of_the_world" "Society_and_social_sciences" "Society_and_social_sciences__Social_sciences" "Society_and_social_sciences__Society" "Technology_and_applied_sciences" "Technology_and_applied_sciences__Agriculture" "Technology_and_applied_sciences__Computing" "Technology_and_applied_sciences__Engineering" "Technology_and_applied_sciences__Transport" "alg-geom" "ao-sci" "astro-ph" "astro-ph.CO" "astro-ph.EP" "astro-ph.GA" "astro-ph.HE" "astro-ph.IM" "astro-ph.SR" "astro-ph_l1" "atom-ph" "bayes-an" "chao-dyn" "chem-ph" "cmp-lg" "comp-gas" "cond-mat" "cond-mat.dis-nn" "cond-mat.mes-hall" "cond-mat.mtrl-sci" "cond-mat.other" "cond-mat.quant-gas" "cond-mat.soft" "cond-mat.stat-mech" "cond-mat.str-el" "cond-mat.supr-con" "cond-mat_l1" "cs.AI" "cs.AR" "cs.CC" "cs.CE" "cs.CG" "cs.CL" "cs.CR" "cs.CV" "cs.CY" "cs.DB" "cs.DC" "cs.DL" "cs.DM" "cs.DS" "cs.ET" "cs.FL" "cs.GL" "cs.GR" "cs.GT" "cs.HC" "cs.IR" "cs.IT" "cs.LG" "cs.LO" "cs.MA" "cs.MM" "cs.MS" "cs.NA" "cs.NE" "cs.NI" "cs.OH" "cs.OS" "cs.PF" "cs.PL" "cs.RO" "cs.SC" "cs.SD" "cs.SE" "cs.SI" "cs.SY" "cs_l1" "dg-ga" "econ.EM" "econ.GN" "econ.TH" "econ_l1" "eess.AS" "eess.IV" "eess.SP" "eess.SY" "eess_l1" "eval_sets" "funct-an" "gr-qc" "hep-ex" "hep-lat" "hep-ph" "hep-th" "math-ph" "math.AC" "math.AG" "math.AP" "math.AT" "math.CA" "math.CO" "math.CT" "math.CV" "math.DG" "math.DS" "math.FA" "math.GM" "math.GN" "math.GR" "math.GT" "math.HO" "math.IT" "math.KT" "math.LO" "math.MG" "math.MP" "math.NA" "math.NT" "math.OA" "math.OC" "math.PR" "math.QA" "math.RA" "math.RT" "math.SG" "math.SP" "math.ST" "math_l1" "mtrl-th" "nlin.AO" "nlin.CD" "nlin.CG" "nlin.PS" "nlin.SI" "nlin_l1" "nucl-ex" "nucl-th" "only_text" "only_txt2" "patt-sol" "physics.acc-ph" "physics.ao-ph" "physics.app-ph" "physics.atm-clus" "physics.atom-ph" "physics.bio-ph" "physics.chem-ph" "physics.class-ph" "physics.comp-ph" "physics.data-an" "physics.ed-ph" "physics.flu-dyn" "physics.gen-ph" "physics.geo-ph" "physics.hist-ph" "physics.ins-det" "physics.med-ph" "physics.optics" "physics.plasm-ph" "physics.pop-ph" "physics.soc-ph" "physics.space-ph" "physics_l1" "plasm-ph" "q-alg" "q-bio" "q-bio.BM" "q-bio.CB" "q-bio.GN" "q-bio.MN" "q-bio.NC" "q-bio.OT" "q-bio.PE" "q-bio.QM" "q-bio.SC" "q-bio.TO" "q-bio_l1" "q-fin.CP" "q-fin.EC" "q-fin.GN" "q-fin.MF" "q-fin.PM" "q-fin.PR" "q-fin.RM" "q-fin.ST" "q-fin.TR" "q-fin_l1" "quant-ph" "solv-int" "stat.AP" "stat.CO" "stat.ME" "stat.ML" "stat.OT" "stat.TH" "stat_l1" "supr-con")
# Loop through each dataset name
checkpoint="gpt2-large_task1/checkpoint-3500"
length=${#datasets[@]}
for idx in {2..239}; do
  dataset_name="${datasets[$idx]}"
  echo "Checkpoint: $checkpoint"
  echo "Currently training: $dataset_name"
  # Run the command with the dataset name
  deepspeed --num_gpus=1 run_clm.py \
  --deepspeed ds_config.json \
  --model_name_or_path gpt2-large \
  --dataset_name "machelreid/m2d2" \
  --resume_from_checkpoint "$checkpoint" \
  --dataset_config_name "$dataset_name" \
  --do_train \
  --do_eval \
  --fp16 \
  --evaluation_strategy="steps" \
  --output_dir "gpt2-large_task$idx" \
  --eval_steps 200 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64
  
  checkpoint="gpt2-large_task$idx"
  cd "$checkpoint"
  variable=$(ls -td -- */ | head -n 1)
  checkpoint="${checkpoint}/${variable}"
  cd ../
  python3 validate_m2d2_continual.py --model_path "$checkpoint"
  #rm -rf "gpt2-large_Task$((idx -1))/checkpoint-500"
  #if [ $idx -gt 0 ]; then
  #  rm -rf "gpt2-large_task$((idx - 1))"
  # fi
done
