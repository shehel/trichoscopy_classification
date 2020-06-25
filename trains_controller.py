from trains import Task
import os
param = {}
# Connect the hyper-parameter dictionary to the task

param["Task Name"] = "Basic Augmentation"
# In this example we pass next task's name as a parameter
param['1'] = 'Data Preprocess'
param['2'] = 'Train Eval'

# The queue where we want the template task (clone) to be sent to
param['execution_queue_name'] = 'default'

# Initialize the task pipelines first Task used to start the pipeline
task = Task.init(task_name=param["Task Name"])

param = task.connect(param)
# Data folder we want to track
task.upload_artifact('local data folder', artifact_object=os.path.join('data'))

# Get a reference to the task to pipe to.
next_task = Task.get_task(project_name=task.get_project_name(), task_name=param["1"])

# Clone the task to pipe to. This creates a task with status Draft whose parameters can be modified.
cloned_task = Task.clone(source_task=next_task, name='Auto generated data prep task')
print('Enqueue next step in pipeline to queue: {}'.format(param['execution_queue_name']))
Task.enqueue(cloned_task.id, queue_name=param['execution_queue_name'])

# Get a reference to the task to pipe to.
next_task = Task.get_task(project_name=task.get_project_name(), task_name=param["2"])

# Clone the task to pipe to. This creates a task with status Draft whose parameters can be modified.
cloned_task = Task.clone(source_task=next_task, name='Auto generated train eval task')
print('Enqueue next step in pipeline to queue: {}'.format(param['execution_queue_name']))
Task.enqueue(cloned_task.id, queue_name=param['execution_queue_name'])

print("Execution Finished.")



