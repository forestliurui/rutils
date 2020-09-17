import os
import subprocess
from .gpu import get_free_gpus_from_hostlist

class Job:
    def __init__(self):
        self.lsf_filename = None
        self.host_list = None # used for -hostfile option for bsub
        self.queue = "gpu_p100" # -q option; specify the submission queue
        self.walltime = None # -W option; specify the max wall time duration in hours
        self.cwd = None
        self.job_name = None # -J option; job name
        self.span_ptile = None # -R "span[ptile=3]"
        self.num_jobs = None # -n option; no need if host_list is used
        self.m = None # -m "gpu-cn012 gpu-cn013"; no need if host_list is used
        self.error_filename = None # -e option
        self.output_filename = None # -o option
        self.email_notification = "ruixliu@umich.edu" # -u option; specify the email
        self.base_dir = "/gpfs/gpfs0/groups/mozafari/ruixliu/tmp/lsf/"
        
        self.command = None        
    
    def generate_hostfile(self, host_filename="hostfile"):
        if self.host_list is not None:
            self.host_filename =  os.path.join(self.base_dir, host_filename)
            with open(self.host_filename, "w") as hfile:
                for hostname in self.host_list:
                    hfile.write("{}\n".format(hostname))
        return self.host_filename
    
    def generate_lsffile(self, lsf_filename="lsffile"):
        self.lsf_filename = os.path.join(self.base_dir, lsf_filename)
        with open(self.lsf_filename, "w") as lfile:
            lfile.write("#!/bin/bash\n\n")
            lfile.write("#BSUB -a openmpi\n")
            
            if self.walltime is not None:
                lfile.write("#BSUT -W {}:0\n".format(self.walltime))
            
            if self.lsf_filename is None and self.num_jobs is not None:
                lfile.write("#BSUB -n {}\n".format(self.num_jobs))
            
            if self.lsf_filename is None and self.m is not None:
                m_string = " ".join(self.m)
                lfile.write("#BSUB -m \"{}\"\n".format(m_string))
                
            if self.span_ptile is not None:
                lfile.write("#BSUB -R \"span[ptile={}]\"\n".format(self.span_ptile))
                
            if self.queue is not None:
                lfile.write("#BSUB -q {}\n".format(self.queue))
                
            if self.job_name is not None:
                lfile.write("#BSUB -J {}\n".format(self.job_name))

            if self.cwd is not None:
                lfile.write("#BSUB -cwd {}\n".format(self.cwd))
                
            error_filepath = os.path.join(self.base_dir, "errors.%J")
            lfile.write("#BSUB -e {}\n".format(error_filepath))
            
            output_filepath = os.path.join(self.base_dir, "output.%J")
            lfile.write("#BSUB -o {}\n".format(output_filepath))
            
            lfile.write("\n")
            lfile.write(self.command)
            lfile.write("\n")
        return self.lsf_filename
    
    def submit(self):
        command_line = []
        command_line.append("bsub")
        
        if self.host_list is not None:
            self.generate_hostfile()
            command_line.append("-hostfile")
            command_line.append(self.host_filename)
            
        self.generate_lsffile()
        command_line.append("<{}".format(self.lsf_filename))
        
        status = subprocess.Popen(" ".join(command_line), shell=True, stdout=subprocess.PIPE)
        print("submit job with command:")
        print(" ".join(command_line))
        print(status.stdout.read())


def get_avail_slots():
    bhosts_output = subprocess.Popen("bhosts |grep gpu |grep ok", shell=True, stdout=subprocess.PIPE).stdout.read().decode("utf-8")
    bhosts_split = bhosts_output.split("\n")
    host_cpu_avail_dict = {}
    for row in bhosts_split[:-1]:
        
        row_split = row.split(" ")
        row_list = []
        for temp in row_split:
            if temp != "":
                row_list.append(temp)
        host_cpu_avail_dict[row_list[0]] = int(row_list[3]) - int(row_list[4])
    
    host_gpu_avail_dict = get_free_gpus_from_hostlist(list(host_cpu_avail_dict.keys()))
    host_avail_dict = {
                        host: min(host_cpu_avail_dict[host], len(host_gpu_avail_dict[host])) 
                        for host in host_cpu_avail_dict.keys()
    }
    return host_avail_dict, host_gpu_avail_dict


def select_hosts(budget=1, host_avail_dict=None):
    selected = []
    if host_avail_dict is None:
        host_avail_dict, _ = get_avail_slots()
    avail = dict(host_avail_dict)
    while budget >0:
        for host in avail.keys():
            if budget >0 and avail[host] > 0:
                avail[host] -= 1
                budget -= 1
                selected.append(host)
    return sorted(selected)