#!/bin/bash

# Define the list of dataset names
dataset_l2_wiki=("Culture_and_the_arts__Culture_and_Humanities" "Culture_and_the_arts__Games_and_Toys" "Culture_and_the_arts__Mass_media" "Culture_and_the_arts__Performing_arts" "Culture_and_the_arts__Sports_and_Recreation" "Culture_and_the_arts__The_arts_and_Entertainment" "Culture_and_the_arts__Visual_arts" "General_referece__Further_research_tools_and_topics" "General_referece__Reference_works" "Health_and_fitness__Exercise" "Health_and_fitness__Health_science" "Health_and_fitness__Human_medicine" "Health_and_fitness__Nutrition" "Health_and_fitness__Public_health" "Health_and_fitness__Self_care" "History_and_events__By_continent" "History_and_events__By_period" "History_and_events__By_region" "Human_activites__Human_activities" "Human_activites__Impact_of_human_activity" "Mathematics_and_logic__Fields_of_mathematics" "Mathematics_and_logic__Logic" "Mathematics_and_logic__Mathematics" "Natural_and_physical_sciences__Biology" "Natural_and_physical_sciences__Earth_sciences" "Natural_and_physical_sciences__Nature" "Natural_and_physical_sciences__Physical_sciences" "Philosophy_and_thinking__Philosophy" "Philosophy_and_thinking__Thinking" "Religion_and_belief_systems__Allah" "Religion_and_belief_systems__Belief_systems" "Religion_and_belief_systems__Major_beliefs_of_the_world" "Society_and_social_sciences__Social_sciences" "Society_and_social_sciences__Society" "Technology_and_applied_sciences__Agriculture" "Technology_and_applied_sciences__Computing" "Technology_and_applied_sciences__Engineering" "Technology_and_applied_sciences__Transport" )
# Loop through each dataset name
checkpoint="gpt2-large_task1/checkpoint-3500"
length=${#dataset_l2_wiki[@]}
for idx in {0..239}; do
  dataset_name="${dataset_l2_wiki[$idx]}"
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
