from remote_run import sync
import logging
import time
from time import gmtime, strftime, localtime
from qd_common import init_logging
from multiprocessing import Process
from email_util import notify_by_email


def backup_all():
    ssh_infos = []
    #ssh_infos.append({'username': 'jianfw',
            #'ip': '157.54.144.79'})
    ssh_infos.append({'username':'REDMOND.jianfw', 
            'ip':'vig-gpu01'})
    #ssh_infos.append({'username':'REDMOND.jianfw', 
            #'ip':'vig-gpu02'})

    backup_list = ['quickdetection', 
            'mysite']
    #backup_list = ['quickdetection']
    for ssh_info in ssh_infos:
        for b in backup_list:
            sync(ssh_info=ssh_info, from_folder='/home/jianfw/code/{}/'.format(b), 
                    target_folder='~/code/{}_backup/'.format(b),
                    delete=True)

def sync_one_time():
    gpu2 = {'username':'REDMOND.jianfw', 
                'ip':'vig-gpu02.guest.corp.microsoft.com'}
    sync(gpu2)

if __name__ == '__main__':
    init_logging()
    while True:
        p = Process(target=backup_all)
        p.start()
        start_time = time.time()
        while True:
            if p.is_alive():
                if time.time() - start_time > 60 * 5:
                    logging.info('time elapsed. will terminate it')
                    notify_by_email('sync process failed', '')
                    p.terminate()
                    break
                else:
                    logging.info('waiting to sync')
                    time.sleep(5)
            else:
                break
        p.join()
        if p.exitcode != 0:
            notify_by_email('sync process does not return 0',
                    'code-{}'.format(p.exitcode))
        logging.info('waiting before to start next')
        time.sleep(60 * 10)
    notify_by_email('sync process returned', '')

