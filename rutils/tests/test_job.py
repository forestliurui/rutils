from ..job import Job

job = Job()
job.host_list = ["gpu-cn014", "gpu-cn015"] # select_hosts(2)  # 
job.job_name = "test_jupyter"
job.cwd = "/gpfs/gpfs0/groups/mozafari/ruixliu/code/distributed_learning/pytorch/scripts"
job.command = "mpirun python test_env.py"

job.submit()

