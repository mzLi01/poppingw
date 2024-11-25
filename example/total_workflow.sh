#!/bin/bash
nnodes=64

result_dir='result'
events_infer_path=${result_dir}/events_infer.h5
events_select_path=${result_dir}/events_select.h5
true_pop_path=${result_dir}/pop.npz
likelihood_dir=${result_dir}/likelihood
range_path=${likelihood_dir}/range.npz
likelihood_path=${likelihood_dir}/plk.npz
logl_path=${likelihood_dir}/logl.npz
mkdir -p ${likelihood_dir}

infer_result_dir=${result_dir}/infer

echo $(date "+%Y-%m-%d %H:%M:%S") 'start generating events'
python sample_events.py --nevents 10000 --outpath ${events_infer_path}
python sample_events.py --nevents 100000 --outpath ${events_select_path}

echo $(date "+%Y-%m-%d %H:%M:%S") 'start calculating likelihood'
mpiexec -np ${nnodes} python match_filter_likelihood.py --events-path ${events_infer_path} --range-path ${range_path} \
    --logl-path ${logl_path} --likelihood-path ${likelihood_path} \
    --netfile detectors/ET_2CE_gwfast.json

echo $(date "+%Y-%m-%d %H:%M:%S") 'start population inference'
python infer_spl.py --likelihood-path ${likelihood_path} --true-pop-path ${true_pop_path} \
    --events-infer-path ${events_infer_path} --events-select-path ${events_select_path} \
    --outdir ${infer_result_dir} --npool ${nnodes}
echo $(date "+%Y-%m-%d %H:%M:%S") 'end'