python generate_vcoco_official.py \
    --param_path pretrained/cql_vcoco.pth \
    --save_path logs/vcoco_cql_eval/vcoco.pkl \
    --hoi_path data/v-coco \
    --backbone resnet50 \
    --interaction_decoder \
    --image_verb_loss \
    --num_workers 6 \
    --verb_embed_norm \
    --cat_specific_fc \
