import os
import sys
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="??")
    parser.add_argument(
        "--path_storage",
        type=str,
        default=None,
        required=True,
        help="Path to storage.",
    )
    parser.add_argument(
        "--path_shell",
        type=str,
        default=None,
        required=True,
        help="Path to the shell file that will be executed in the job.",
    )
    parser.add_argument(
        "--path_repo",
        type=str,
        default=None,
        required=True,
        help="Path to local LAVIS repo.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main():
    args = parse_args()
    # TODO considerar el log nuevo
    # logging.basicConfig(filename='train_blip2.log', level=logging.INFO)
    # logging.basicConfig(filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    # logging.info('Started')
    print('Running train_blip2.py')
    # wandb.login()
    # configure environment variables
    try: 
        storage_path = args.path_storage
        # storage_path = '/home/fpcattan/nas2_grima/'
        # env_path = os.path.join(storage_path, '.pyenv/versions/lavis_env/')
        shell_path = os.path.join(storage_path, args.path_shell)
        # print('Storage path:',storage_path)
        # print('Shell path:', shell_path)
        # shell_path = os.path.join(storage_path, 'LAVIS_nav/run_scripts/blip2/train/pretrain_stage1.sh')
        # shell_path = os.path.join(storage_path, 'LAVIS/run_scripts/blip2/train/pretrain_stage2.sh')
        # train_shell_path = 'run_scripts/blip2/train/pretrain_stage1.sh'
        # train_blip2_path = 'run_scripts/blip2/train/pretrain_stage2.sh'

        os.environ['TRANSFORMERS_CACHE'] = os.path.join(storage_path, '.cache/')
        os.environ['HF_HOME'] = os.path.join(storage_path, '.cache/huggingface/')
        # print('TRANSFORMERS_CACHE:',os.environ['TRANSFORMERS_CACHE'])
        # print('HF_HOME:',os.environ['HF_HOME'])
        # os.environ['XDG_CACHE_HOME'] = os.path.join(storage_path, '.cache/')
        
        repo_path = os.path.join(storage_path, args.path_repo)
        # print('Repo path:',repo_path)
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
        
        # TRAININGG
        os.system('/bin/bash ' + shell_path) # ejecuta el proceso
        
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
