import os
import sys
import logging

# import torch

def main():
    # TODO considerar el log nuevo
    # logging.basicConfig(filename='train_blip2.log', level=logging.INFO)
    # logging.basicConfig(filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    # logging.info('Started')
    print('Running train_blip2.py')
    # wandb.login()
    # configure environment variables
    try: 
        storage_path = '/home/fpcattan/storage/'
        env_path = os.path.join(storage_path, '.pyenv/versions/blip2/')
        shell_path = os.path.join(storage_path, 'LAVIS_nav/run_scripts/blip2/train/pretrain_stage1.sh')
        # shell_path = os.path.join(storage_path, 'LAVIS/run_scripts/blip2/train/pretrain_stage2.sh')
        # train_shell_path = 'run_scripts/blip2/train/pretrain_stage1.sh'
        # train_blip2_path = 'run_scripts/blip2/train/pretrain_stage2.sh'

        os.environ['TRANSFORMERS_CACHE'] = os.path.join(storage_path, '.cache/')
        os.environ['HF_HOME'] = os.path.join(storage_path, '.cache/huggingface/')
        os.environ['XDG_CACHE_HOME'] = os.path.join(storage_path, '.cache/')
        
        repo_path = os.path.join(storage_path, 'LAVIS_nav/')
        sys.path.append(repo_path)

        # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        #print('Device: ', device)
        # logging.info('Device: ', device)
        # logging.info('Number of devices: ', torch.cuda.device_count())
        # print('Number of devices: ', str(torch.cuda.device_count()))
        # clean cache
        # torch.cuda.empty_cache()
        # subprocess.run(['bash', train_shell_path])
        # res = subprocess.run(['pwd'], check=True)
        # print('res: ', res)
        # os.system('/bin/bash ' + train_shell_path)
        os.system('/bin/bash ' + shell_path)
        #print('Executed the shell of training!')
        # logging.info('Executed the shell of training!')
    except Exception as e:
        logging.error('Error executing: ', e)
        # print('Exception', e)
    finally:
        logging.info('End of training!!!')
        print('Finished train_blip2.py')


if __name__ == '__main__':
    main()
