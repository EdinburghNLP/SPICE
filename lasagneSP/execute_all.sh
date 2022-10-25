set -e
QTYPE_ARRAY_WITHOUT_CONTEXT=("Clarification" "Comparative Reasoning (All)" "Comparative Reasoning (Count) (All)" "Logical Reasoning (All)" "Quantitative Reasoning (All)" "Quantitative Reasoning (Count) (All)" "Simple Question (Direct)" "Simple Question (Ellipsis)")
QTYPE_ARRAY_WITH_CONTEXT=("Simple Question (Coreferenced)" "Verification (Boolean) (All)")
base_p="prefix_path_test_"
split=0
out_dir="outdir"
mkdir ${out_dir}

for val in "${QTYPE_ARRAY_WITHOUT_CONTEXT[@]}"; do
    echo $val
    python run_subtype_lf.py --file_path "${base_p}${val}"".json" --question_type "${val}" --server_link "http://127.0.0.1:9999/blazegraph/namespace/wd/sparq" --out_eval_file split1_"${val}"_intermediate.json > ${out_dir}/split${split}_"${val}"_intermediate.out 2> ${out_dir}/split${split}_"${val}"_intermediate.out 
done

for val in "${QTYPE_ARRAY_WITH_CONTEXT[@]}"; do
    echo $val
    python run_subtype_lf.py --file_path "${base_p}${val}"".json" --question_type "${val}" --server_link "http://127.0.0.1:9999/blazegraph/namespace/wd/sparq" --context_dist_file CSQA_v9_skg.v6_compar_spqres9_subkg2_tyTop_nelctx_cleaned_context_distance_test.log --out_eval_file split1_"${val}"_intermediate.json > ${out_dir}/split${split}_"${val}"_intermediate.out 2> ${out_dir}/split${split}_"${val}"_intermediate.out
done
