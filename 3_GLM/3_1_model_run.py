
# coding: utf-8

# # Subject-level modeling
# Natalia Vélez, February 2019
# 
# This script does run- and subject-level modeling, using outputs from fmriprep. Our workflow is based on [this script](https://github.com/poldrack/fmri-analysis-vm/blob/master/analysis/postFMRIPREPmodelling) from the Poldrack Lab repository.

# Load libraries:

# In[1]:


from IPython.display import Image # Debug

import os  # system functions
import sys
import pandas as pd
import glob
import numpy as np
import json
from os.path import join as opj
import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
from nipype.interfaces import utility as niu  # Utilities
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as modelgen  # model generation
import nipype.algorithms.rapidart as ra  # artifact detection

from nipype.workflows.fmri.fsl import create_susan_smooth

from nipype import config
config.enable_debug_mode()


# Project directory and function inputs:

# In[2]:


# For testing (COMMENT)
#project = 'SwiSt'
#subject = 'sub-06'
#task = 'tomloc'
#model = 'localizer'
#n_runs = 2
_, project, subject, task, model, n_runs = sys.argv
n_runs = int(n_runs)
print(project)
print(subject)
print(task)
print(model)
print(n_runs)
runs = list(range(1, n_runs+1))

# Location of project
scratch_dir = os.environ['PI_SCRATCH']
project_dir = opj(scratch_dir, project, 'BIDS_data')
derivatives_dir = opj(project_dir, 'derivatives')
work_dir = opj(scratch_dir, project, 'cache', 'l1', 'task-%s_model-%s_%s' %(task, model, subject))
print(project_dir)

# Task repetition time:

# In[3]:


task_info_file = opj(project_dir, 'task-%s_bold.json' % task)

# Load task info
with open(task_info_file, 'r') as f:
    task_info = json.load(f)

# Get TR from task info
TR = task_info['RepetitionTime']
print('TR: %.02f' % TR) # DEBUG


# ## Specify model

# **IdentityInterface:** Iterate over subjects and runs

# In[4]:


inputnode = pe.Node(niu.IdentityInterface(fields=['project', 'subject_id', 'task', 'model', 'run'],
                                         mandatory_inputs=True),
                   'inputnode')
inputnode.iterables = [('run', runs)]
inputnode.inputs.project = project
inputnode.inputs.subject_id = subject
inputnode.inputs.task = task
inputnode.inputs.model = model


# **DataGrabber:** Select files

# In[12]:


# Templates for DataGrabber
func_template = 'smooth_fmriprep/%s/func/*task-%s_run-%02d_space-MNI152NLin2009cAsym_desc-smoothed_bold.nii.gz'
anat_template = 'fmriprep/%s/anat/*space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
mask_template = 'fmriprep/%s/func/*task-%s_run-%02d_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
confounds_template = 'fmriprep/%s/func/*task-%s_run-%02d_desc-confounds_regressors.tsv'
model_template = 'l1model/%s/func/*task-%s_model-%s_run-%02d_events.tsv'
contrast_template = 'l1model/task-%s_model-%s.json'


# In[13]:


datasource = pe.Node(nio.DataGrabber(infields=['subject_id',
                                               'task',
                                               'model',
                                               'run'],
                                    outfields=['struct',
                                               'func',
                                               'mask',
                                               'confounds_file',
                                               'events_file',
                                               'contrasts_file']),
                    'datasource')

datasource.inputs.base_directory = derivatives_dir
datasource.inputs.template = '*'
datasource.inputs.sort_filelist = True
datasource.inputs.field_template = dict(struct=anat_template,
                                       func=func_template,
                                       mask=mask_template,
                                       confounds_file=confounds_template,
                                       events_file=model_template,
                                       contrasts_file=contrast_template)
datasource.inputs.template_args = dict(struct=[['subject_id']],
                                      func=[['subject_id', 'task', 'run']],
                                      mask=[['subject_id', 'task', 'run']],
                                      confounds_file=[['subject_id', 'task', 'run']],
                                      events_file=[['subject_id', 'task', 'model', 'run']],
                                      contrasts_file=[['task', 'model']])


# **RapidArt:** Identify motion outliers

# Helper function: Load motion parameters from fmriprep

# In[14]:


def get_mcparams(confounds_file):
    import os
    import pandas as pd
    pd.read_csv(confounds_file, sep='\t').to_csv(
        'mcparams.tsv', sep='\t',
        columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
        header=False, index=False)
    return os.path.abspath('mcparams.tsv')

mcparams = pe.Node(niu.Function(input_names=['confounds_file'],
                                output_names=['realignment_parameters'],
                                     function=get_mcparams),
                        'mcparams')


# Rapidart node

# In[15]:


art = pe.Node(ra.ArtifactDetect(), 'art')
art.inputs.parameter_source = 'SPM'
art.inputs.norm_threshold = 1
art.inputs.use_differences = [True, False]
art.inputs.zintensity_threshold = 3
art.inputs.mask_type = 'file'


# **ModelGrabber:** Grab model specification info (util)

# In[16]:
def ModelGrabber(contrasts_file, events_file, confounds_file):

    from os import environ
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base import Bunch
    from os.path import join as opj
    from json import load as loadjson

    # Project dir
    project = 'SwiSt'
    project_dir = opj(environ['PI_SCRATCH'], project, 'BIDS_data')

    ### Load data ###
    read_tsv = lambda f: pd.read_csv(opj(project_dir, f), sep='\t', index_col=None)
    model = read_tsv(events_file)
    all_confounds = read_tsv(confounds_file)

    ### Confounds ###
    confound_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                      'global_signal', 'framewise_displacement',
                      'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
                      'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05']
    confounds_df = all_confounds.loc[:,confound_names]
    confounds_df.framewise_displacement = confounds_df.framewise_displacement.fillna(0)
    confounds_dict = confounds_df.to_dict('list')

    # Convert confounds to dictionary
    confounds = {'regressor_names': confound_names,
                'regressors': [confounds_dict[name] for name in confound_names]}

    ### Model specification ###
    modelspec_dict = model.copy().rename(index=str,
                                         columns={'trial_type': 'conditions',
                                                  'onset': 'onsets',
                                                  'duration': 'durations',
                                                  'amplitude': 'amplitudes'})
    modelspec_dict = modelspec_dict.groupby('conditions').aggregate(lambda g: list(g)).reset_index().to_dict('list')
    modelspec_dict.update(confounds)
    modelspec_dict['amplitudes'] = [a if not all(np.isnan(a)) else np.ones(np.size(a)) for a in modelspec_dict['amplitudes']]
    modelspec = Bunch(**modelspec_dict)
    
    ### Contrasts ###
    with open(opj(project_dir, contrasts_file), 'r') as contrast_handle:
        contrasts = loadjson(contrast_handle)

    return modelspec, contrasts


# Make node
model_grabber = pe.Node(niu.Function(input_names=['contrasts_file', 'events_file', 'confounds_file'],
                                     output_names=['modelspec', 'contrasts'],
                                     function=ModelGrabber),
                        'model_grabber')


# **ModelSpec**: Model specification

# In[17]:


modelspec = pe.Node(modelgen.SpecifyModel(),
                   'modelspec')
modelspec.inputs.time_repetition = TR
modelspec.inputs.input_units = 'secs'
modelspec.inputs.high_pass_filter_cutoff = 128.0


# **level1design:** Generate FEAT-specific files

# In[18]:


level1design = pe.Node(fsl.model.Level1Design(),
                  'l1design')
level1design.inputs.bases = {'dgamma':{'derivs': True}}
level1design.inputs.model_serial_correlations = True
level1design.inputs.interscan_interval = TR


# ## Estimate model & contrasts

# **FEATModel:** Prepare design file for first-level model

# In[19]:


featmodel = pe.Node(fsl.model.FEATModel(),
                   'featmodel')


# **ApplyMask:** Prepare brainmask for modeling

# In[20]:


mask = pe.Node(fsl.maths.ApplyMask(),
              'mask')


# **FILM:** Run-specific model

# In[21]:


filmgls = pe.Node(fsl.FILMGLS(),
                  'filmgls')
filmgls.inputs.autocorr_noestimate = True


# ## Subject-level fit

# Helper function: Sort FILM outputs

# In[22]:


pass_run_data = pe.Node(niu.IdentityInterface(fields = ['mask', 'dof_file', 'copes', 'varcopes']), 'pass_run_data')

join_run_data = pe.JoinNode(
        niu.IdentityInterface(fields=['masks', 'dof_files', 'copes', 'varcopes']),
        joinsource='inputnode',
        joinfield=['masks', 'dof_files', 'copes', 'varcopes'],
        name='join_run_data')

def sort_filmgls_output(copes_grouped_by_run, varcopes_grouped_by_run):
    
    def reshape_lists(files_grouped_by_run):
        import numpy as np
        if not isinstance(files_grouped_by_run, list):
            files = [files_grouped_by_run]
        else:
            files = files_grouped_by_run
            
        if all(len(x) == len(files[0]) for x in files):
            n_files = len(files[0])
        else:
            ('{}DEBUG - files {}'.format('-=-', len(files)))
            print(files)

        all_files = np.array(files).flatten()
        files_grouped_by_contrast = all_files.reshape(int(len(all_files) / n_files), n_files).T.tolist()
        
        return files_grouped_by_contrast
    
    copes_grouped_by_contrast = reshape_lists(copes_grouped_by_run)
    varcopes_grouped_by_contrast = reshape_lists(varcopes_grouped_by_run)
    
    print('{}DEBUG - copes_grouped_by_contrast {}'.format('==-', len(copes_grouped_by_contrast)))
    print(copes_grouped_by_contrast)
    
    print('{}DEBUG - varcopes_grouped_by_contrast {}'.format('---', len(varcopes_grouped_by_contrast)))
    print(varcopes_grouped_by_contrast)
    
    return copes_grouped_by_contrast, varcopes_grouped_by_contrast


group_by_contrast = pe.Node(niu.Function(input_names=['copes_grouped_by_run',
                                                      'varcopes_grouped_by_run'],
                                         output_names=['copes_grouped_by_contrast',
                                                       'varcopes_grouped_by_contrast'],
                                         function=sort_filmgls_output), name='group_by_contrast')


pickfirst = lambda x: x[0]

num_copes = lambda x: len(x)


# Level 2 model:

# In[23]:


l2model = pe.Node(fsl.model.L2Model(),
                 'l2model')


# Merge copes and varcopes:

# In[24]:


copemerge = pe.MapNode(
    interface=fsl.Merge(dimension='t'),
    iterfield=['in_files'],
    name="copemerge")

varcopemerge = pe.MapNode(
    interface=fsl.Merge(dimension='t'),
    iterfield=['in_files'],
    name="varcopemerge")


# Create DOF file:

# In[25]:


def dof_vol_fun(dof_files, cope_files):
    import os
    import nibabel as nb
    import numpy as np
    
    img = nb.load(cope_files[0])
    n_runs = len(cope_files)
    out_data = np.zeros(list(img.shape) + [n_runs])
    for i in range(out_data.shape[-1]):
        dof = np.loadtxt(dof_files[i])
        out_data[:, :, :, i] = dof
    filename = os.path.join(os.getcwd(), 'dof_file.nii.gz')
    newimg = nb.Nifti1Image(out_data, None, img.header)
    newimg.to_filename(filename)
    
    return filename

dof_vol = pe.MapNode(
    niu.Function(
        input_names=['dof_files', 'cope_files'],
        output_names=['dof_volume'],
        function=dof_vol_fun),
    iterfield=['cope_files'],
    name='dof_vol')


# Merge outputs by cope:

# In[26]:


pass_fixedfx_inputs = pe.JoinNode(niu.IdentityInterface(fields=['copes', 'varcopes', 'dof_volume']),
                                  joinfield=['copes', 'varcopes', 'dof_volume'],
                                  joinsource='inputnode',
                                  name='pass_fixedfx_inputs')


# **DataSink:** Specify outputs of first-level modeling workflow

# In[29]:


datasink = pe.Node(nio.DataSink(),
                  'datasink')
datasink.inputs.base_directory = opj(derivatives_dir, 'model', 'task-%s' % task, 'model-%s' % model,  subject)
datasink.inputs.substitutions = [('_run_', 'run-0'),
                                 ('run0', 'design')]
datasink.inputs.regexp_substitutions = [('_dof_vol[0-9]+\/', ''),
                                        ('_copemerge[0-9]+\/', ''),
                                        ('_varcopemerge[0-9]+\/', ''),
					('sub-[0-9]+_task-[A-Za-z0-9]+_run-[0-9]+_space-[A-Za-z0-9]+_desc-brain_mask', 'brain_mask')]


# **l1_workflow:** Build and run first-level modeling workflow

# In[32]:


level1_workflow = pe.Workflow('l1', base_dir = work_dir)

level1_workflow.connect([
    ### Build first level model
    (inputnode, datasource, [
        ('subject_id', 'subject_id'),
        ('task', 'task'),
        ('model', 'model'),
        ('run', 'run')]),
    (inputnode, model_grabber, [
       ('project', 'project')]),
    (datasource, model_grabber, [
        ('contrasts_file', 'contrasts_file'),
        ('events_file', 'events_file'),
        ('confounds_file', 'confounds_file')]),
    (datasource, mcparams, [
        ('confounds_file', 'confounds_file')
    ]),
    (datasource, art, [
        ('mask', 'mask_file'),
        ('func', 'realigned_files')
    ]),
    (mcparams, art, [
        ('realignment_parameters', 'realignment_parameters')
    ]),
    (datasource, modelspec, [('func', 'functional_runs')]),
    (model_grabber, modelspec, [('modelspec', 'subject_info')]),
    (art, modelspec, [('outlier_files', 'outlier_files')]),
    (model_grabber, level1design, [('contrasts', 'contrasts')]),
    (modelspec, level1design, [('session_info', 'session_info')]),
    (level1design, featmodel, [
        ('fsf_files', 'fsf_file'),
        ('ev_files', 'ev_files')]),
    (datasource, mask, [
        ('mask', 'mask_file'),
        ('func', 'in_file')
    ]),
    ### Prep functional data
    (mask, filmgls, [('out_file', 'in_file')]),
    
    ### Estimate model
    (featmodel, filmgls, [
        ('design_file', 'design_file'),
        ('con_file', 'tcon_file'),
        ('fcon_file', 'fcon_file')]),
    
    ### Pass inputs for higher-level fits
    (datasource, pass_run_data, [('mask', 'mask')]),
    (filmgls, pass_run_data, [
        ('copes', 'copes'),
        ('varcopes', 'varcopes'),
        ('dof_file', 'dof_file'),
    ]),
    (pass_run_data, join_run_data, [
        ('mask', 'masks'),
        ('dof_file', 'dof_files'),
        ('copes', 'copes'),
        ('varcopes', 'varcopes'),
    ]),
    (join_run_data, group_by_contrast, [
        ('copes', 'copes_grouped_by_run'),
        ('varcopes', 'varcopes_grouped_by_run')
    ]),
    (join_run_data, l2model, [
        (('copes', num_copes), 'num_copes')
    ]),
    (group_by_contrast, copemerge, [
        ('copes_grouped_by_contrast', 'in_files')
    ]),
    (group_by_contrast, varcopemerge, [
        ('varcopes_grouped_by_contrast', 'in_files')
    ]),
    (group_by_contrast, dof_vol, [
        ('copes_grouped_by_contrast', 'cope_files')
    ]),
    (join_run_data, dof_vol, [
        ('dof_files', 'dof_files')
    ]),
    ### Write out model files
    (level1design, datasink, [
        ('fsf_files', '@fsf')
    ]),
    (featmodel, datasink, [('design_file', '@design'),
                           ('design_image', '@design_img'),
                           ('design_cov', '@cov'),
                           ('con_file', '@tcon'),
                           ('fcon_file', '@fcon')]),
    (filmgls, datasink, [
        ('zstats', '@zstats'),
        ('copes', '@copes'),
        ('varcopes', '@varcopes'),
        ('param_estimates', '@parameter_estimates'),
        ('dof_file', '@dof'),
        ('logfile', '@log')
    ]),
    (join_run_data, datasink, [
        (('masks', pickfirst), 'input_fixedfx.@mask')
    ]),
    (l2model, datasink, [
        ('design_mat', 'input_fixedfx.@design_mat'),
        ('design_con', 'input_fixedfx.@design_con'),
        ('design_grp', 'input_fixedfx.@design_grp')
    ]),
    (copemerge, datasink, [
        ('merged_file', 'input_fixedfx.@copes')
    ]),
    (varcopemerge, datasink, [
        ('merged_file', 'input_fixedfx.@varcopes')
    ]),
    (dof_vol, datasink, [
        ('dof_volume', 'input_fixedfx.@dof_volume')
    ]),
    
])


# In[34]:


result = level1_workflow.run()

## QA (Debug Only)

