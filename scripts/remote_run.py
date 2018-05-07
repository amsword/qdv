import subprocess as sp
import re
import sys
import os
from pprint import pprint
import psutil
import time
import yaml
from qd_common import write_to_file
from shutil import copyfile
import os.path as op
import logging
from qd_common import init_logging
from qd_common import is_cluster
from qd_common import ensure_directory

def is_dgx(ssh_info):
    return ssh_info['ip'] == '10.196.44.201' or 'REDMOND' in ssh_info['username'] 

def sync_qd(ssh_info, delete=False):
    target_folder = '/tmp/code/quickdetection/'
    if is_cluster(ssh_info):
        target_folder = '/app/code/quickdetection'
    sync(ssh_info=ssh_info, target_folder=target_folder, delete=delete)
    remote_run('cd {} && ./compile.sh'.format(target_folder), ssh_info)
    remote_run('python -m nltk.downloader all', ssh_info)

    #if is_cluster(ssh_info):
    if not is_dgx(ssh_info):
        remote_run('cd {} && sudo pip install -r requirements.txt'.format(
            target_folder), ssh_info)
    if is_cluster(ssh_info):
        c = []
        c.append('cd {}'.format(op.join(target_folder, 'src/CCSCaffe/python')))
        c.append('sudo python -c "import caffe; caffe.set_mode_gpu()"')
        remote_run(' && '.join(c), ssh_info)

    special_path = ['data', 'output', 'models']
    for p in special_path:
        if p in ssh_info:
            c = []
            c.append('cd {}'.format(target_folder))
            c.append('rm -rf {}'.format(p))
            c.append('ln -s {} {}'.format(ssh_info[p], op.join(target_folder,
                p)))
            remote_run(' && '.join(c), ssh_info)

def sync(ssh_info, from_folder='/home/jianfw/code/quickdetection/', 
        target_folder='/tmp/code/quickdetection/', delete=False):
    remote_run('mkdir -p {}'.format(target_folder), ssh_info)
    cmd = []
    cmd.append('rsync')
    if delete:
        cmd.append('--delete')
    cmd.append('-tvrzh')
    cmd.append('--links')
    def exclude_if_exists(sub):
        if op.exists(op.join(from_folder, sub)):
            cmd.append('--exclude')
            cmd.append('/{}'.format(sub))
    exclude_if_exists('src/CCSCaffe/.build_release/')
    exclude_if_exists('src/CCSCaffe/.build_debug/')
    exclude_if_exists('src/CCSCaffe/.build/')
    exclude_if_exists('src/CCSCaffe/python/caffe/_caffe.so')
    exclude_if_exists('output')
    exclude_if_exists('data')
    exclude_if_exists('models')
    exclude_if_exists('.git')
    exclude_if_exists('src/CCSCaffe/.git')
    exclude_if_exists('tmp_run')
    exclude_if_exists('visualization')
    cmd.append('--exclude')
    cmd.append('*.swp')
    cmd.append('--exclude')
    cmd.append('*.swo')
    cmd.append('--exclude')
    cmd.append('*.caffemodel')
    cmd.append('--exclude')
    cmd.append('*.solverstate')
    cmd.append('--exclude')
    cmd.append('*.pyc')
    cmd.append('{}'.format(from_folder))
    extra_ssh_info = []
    for key in ssh_info:
        if len(key) > 0 and key[0] == '-':
            extra_ssh_info.append(key)
            extra_ssh_info.append(str(ssh_info[key]))
    cmd.append('-e')
    if len(extra_ssh_info) > 0:
        cmd.append('ssh {}'.format(' '.join(extra_ssh_info)))
    else:
        cmd.append('ssh -i /home/jianfw/.ssh/id_rsa')
    cmd.append('{}@{}:{}'.format(ssh_info['username'], 
        ssh_info['ip'],
        target_folder))
    cmd_run(cmd)

def scp_f(local_folder, target_file, ssh_cmd):
    logging.info('ssh info' + str(ssh_cmd))
    logging.info((local_folder, target_file))
    cmd = ['scp', '-r']
    if '-p' in ssh_cmd:
        cmd.append('-P')
        cmd.append(str(ssh_cmd['-p']))
    if '-i' in ssh_cmd:
        cmd.append('-i')
        cmd.append(ssh_cmd['-i'])
    cmd += [local_folder, '{}@{}:{}'.format(ssh_cmd['username'],
        ssh_cmd['ip'], target_file)]
    cmd_run(cmd)

def scp(local_file, target_file, ssh_cmd):
    assert op.isfile(local_file)
    logging.info('ssh info' + str(ssh_cmd))
    logging.info((local_file, target_file))
    cmd = ['scp']
    if '-p' in ssh_cmd:
        cmd.append('-P')
        cmd.append(str(ssh_cmd['-p']))
    if '-i' in ssh_cmd:
        cmd.append('-i')
        cmd.append(ssh_cmd['-i'])
    cmd += [local_file, '{}@{}:{}'.format(ssh_cmd['username'],
        ssh_cmd['ip'], target_file)]
    cmd_run(cmd)

def remote_run(str_cmd, ssh_info, return_output=False):
    cmd = ['ssh', '-t', '-t', '-o', 'StrictHostKeyChecking no']
    for key in ssh_info:
        if len(key) > 0 and key[0] == '-':
            cmd.append(key)
            cmd.append(str(ssh_info[key]))
    cmd.append('{}@{}'.format(ssh_info['username'], ssh_info['ip']))
    if is_cluster(ssh_info):
        prefix = 'source ~/.bashrc && export PATH=/usr/local/nvidia/bin:$PATH && '
    else:
        prefix = 'export PATH=/usr/local/nvidia/bin:$PATH && '
    suffix = ' && hostname'
    cmd.append('{}{}{}'.format(prefix, str_cmd, suffix))

    return cmd_run(cmd, return_output)

def remote_python_run(func, kwargs, ssh_cmd):
    logging.info('ssh_cmd: ' + str(ssh_cmd))
    working_dir = os.path.dirname(os.path.abspath(__file__))
    sudo_cmd = ''
    if is_cluster(ssh_cmd):
        working_dir = '/app/code/quickdetection/scripts'
    working_dir = '/tmp/code/quickdetection/scripts'
        #sudo_cmd = 'sudo '
    # serialize kwargs locally
    if kwargs is None:
        str_args = ''
    else:
        str_args = yaml.dump(kwargs)
    logging.info(str_args)
    param_basename = 'remote_run_param_{}.txt'.format(hash(str_args))
    param_local_file = '/tmp/{}'.format(param_basename)
    write_to_file(str_args, param_local_file)
    param_target_file = op.join(working_dir, param_basename)
    # send it to the remote machine
    scp(param_local_file, param_target_file, ssh_cmd) 
    # generate the script locally
    scripts = []
    scripts.append('import matplotlib')
    scripts.append('matplotlib.use(\'Agg\')')
    scripts.append('from {} import {}'.format(func.__module__,
        func.__name__))
    scripts.append('import yaml')
    scripts.append('param = yaml.load(open(\'{}\', \'r\').read())'.format(
        param_target_file))
    scripts.append('{}(**param)'.format(func.__name__))
    script = '\n'.join(scripts)
    basename = 'remote_run_{}.py'.format(hash(script))
    local_file = '/tmp/{}'.format(basename)
    write_to_file('\n'.join(scripts), local_file)

    target_file = op.join(working_dir, basename)
    scp(local_file, target_file, ssh_cmd)
    remote_run('cd {} && {}python {}'.format(op.dirname(working_dir), 
        sudo_cmd, target_file), ssh_cmd)

def cmd_run2(list_cmd, return_output=False, env=None,
        working_dir=None,
        shell=False):
    logging.info('start to cmd run: {}'.format(' '.join(map(str, list_cmd))))
    # if we dont' set stdin as sp.PIPE, it will complain the stdin is not a tty
    # device. Maybe, the reson is it is inside another process. 
    # if stdout=sp.PIPE, it will not print the result in the screen
    #if env is None:
        #p = sp.Popen(list_cmd, stdin=sp.PIPE, cwd=working_dir)
    #else:
    e = os.environ.copy()
    if env:
        for k in env:
            e[k] = env[k]
    # shell: if true, it means a shell is launched, and we can use * to expand
    # the files, but take care of the command list or command string
    if shell:
        p = sp.Popen(' '.join(list_cmd), 
                stdin=sp.PIPE, 
                env=e, 
                cwd=working_dir,
                shell=shell)
    else:
        p = sp.Popen(list_cmd, 
                stdin=sp.PIPE, 
                env=e, 
                cwd=working_dir,
                shell=shell)
    message = p.communicate()
    if p.returncode != 0:
        raise ValueError(message)
    import ipdb;ipdb.set_trace(context=15)
    
    logging.info('finished the cmd run')
    return message

def cmd_run(list_cmd, return_output=False, env=None,
        working_dir=None,
        shell=False):
    logging.info('start to cmd run: {}'.format(' '.join(map(str, list_cmd))))
    # if we dont' set stdin as sp.PIPE, it will complain the stdin is not a tty
    # device. Maybe, the reson is it is inside another process. 
    # if stdout=sp.PIPE, it will not print the result in the screen
    e = os.environ.copy()
    if working_dir:
        ensure_directory(working_dir)
    if env:
        for k in env:
            e[k] = env[k]
    if not return_output:
        #if env is None:
            #p = sp.Popen(list_cmd, stdin=sp.PIPE, cwd=working_dir)
        #else:
        if shell:
            p = sp.Popen(' '.join(list_cmd), 
                    stdin=sp.PIPE, 
                    env=e, 
                    cwd=working_dir,
                    shell=True)
        else:
            p = sp.Popen(list_cmd, 
                    stdin=sp.PIPE, 
                    env=e, 
                    cwd=working_dir)
        message = p.communicate()
        if p.returncode != 0:
            raise ValueError(message)
    else:
        if shell:
            message = sp.check_output(' '.join(list_cmd), 
                    env=e,
                    cwd=working_dir,
                    shell=True)
        else:
            message = sp.check_output(list_cmd,
                    env=e,
                    cwd=working_dir)
    
    logging.info('finished the cmd run')
    return message

def sync_code():
    assert False, 'use sync()'
    cmd = []
    cmd.append('rsync')
    cmd.append('-vrazh')
    cmd.append('--exclude')
    cmd.append('/src/CCSCaffe/.build_release/*')
    cmd.append('--exclude')
    cmd.append('/src/CCSCaffe/.build_debug/*')
    cmd.append('--exclude')
    cmd.append('*.swp')
    cmd.append('--exclude')
    cmd.append('*.swo')
    cmd.append('--exclude')
    cmd.append('*.pyc')
    cmd.append('--exclude')
    cmd.append('/output')
    cmd.append('./')
    cmd.append('REDMOND.jianfw@vig-gpu01:~/code/quickdetection/')
    cmd_run(cmd)

def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except:
        return False


def collect_process_info():
    result = {}
    for process in psutil.process_iter():
        result[process.pid] = {}
        result[process.pid]['username'] = process.username()
        result[process.pid]['time_spent_in_hour'] = (int(time.time()) -
                process.create_time()) / 3600.0
        result[process.pid]['cmdline'] = ' '.join(process.cmdline())
    return result

if __name__ == '__main__':
    init_logging()
    desktop = {'username': 'jianfw',
                'ip': '157.54.146.114'}
    dl8 = {'username': 'jianfw',
            'ip':'10.196.44.185',
            '-p': 30824,
            '-i': '/vighd/dlworkspace/work/jianfw/.ssh/id_rsa',
            'data': '/work/data/qd_data_cluster', 
            'output': '/work/work/qd_output'}
    vig = {'username':'jianfw', 
            'ip':'10.196.44.201'}
    cluster = {'username': 'jianfw',
            'ip':'viggpu02.redmond.corp.microsoft.com',
            '-p': 30178,
            'models': '/work/work/qd_models',
            'data': '/work/data/qd_data_cluster',
            'output': '/work/work/qd_output'}
    sync_qd(vig, delete=True)

