#include <${default_model}/default_inputs.h>
#include <${default_model}/runtime.h>

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
% endfor
% if golden_cpu_model:
#define GOLDEN_CHECK_BENCH_ITERATIONS ${bench_iterations}
% endif

int main(int argc, char** argv){
    // target specific inits
    % for init_func in target.init_funcs:
    ${init_func}();
    % endfor
    
    match_runtime_ctx match_ctx;

    % for out_name,out in match_outputs.items():
    % if target.alloc_fn != "":
    ${out["c_type"]}* ${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % else:
    ${out["c_type"]} ${out_name}_pt_[${out["prod_shape"]}];
    ${out["c_type"]}* ${out_name}_pt = ${out_name}_pt_;
    % endif
    % if golden_cpu_model:
    % if target.alloc_fn != "":
    ${out["c_type"]}* golden_check_${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % else:
    ${out["c_type"]} golden_check_${out_name}_pt_[${out["prod_shape"]}];
    ${out["c_type"]}* golden_check_${out_name}_pt = golden_check_${out_name}_pt_;
    % endif
    % endif
    % endfor

    match_${"golden_check_" if golden_cpu_model else ""}${default_model}_runtime(
        % for inp_name in match_inputs.keys():
        ${inp_name}_default,
        % endfor
        % if golden_cpu_model:
        % for inp_name in match_inputs.keys():
        ${inp_name}_default,
        % endfor
        % endif
        % for out_name in match_outputs.keys():
        ${out_name}_pt,
        % endfor
        % if golden_cpu_model:
        % for out_name in match_outputs.keys():
        golden_check_${out_name}_pt,
        % endfor
        GOLDEN_CHECK_BENCH_ITERATIONS,
        % endif
        &match_ctx);
    
    % if handle_out_fn!="":
    ${handle_out_fn}(
        % for out_name in match_outputs.keys():
        ${out_name}_pt,
        ${match_outputs[out_name]["prod_shape"]},
        % endfor
        match_ctx.status);
    % endif
    
    % for out_name in match_outputs.keys():
    % if golden_cpu_model and target.free_fn != "":
    ${target.free_fn}(golden_check_${out_name}_pt);
    % endif
    % if target.free_fn != "":
    ${target.free_fn}(${out_name}_pt);
    % endif
    % endfor
    // target specific cleaning functions
    % for clean_func in target.clean_funcs:
    ${clean_func}();
    % endfor
    return 0;
}