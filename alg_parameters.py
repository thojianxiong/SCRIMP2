import datetime

""" Hyperparameters of SCRIMP!"""


class EnvParameters:
    LIFELONG = True
    N_AGENTS = 8  # number of agents used in training
    N_ACTIONS = 5
    EPISODE_LEN = 512 if LIFELONG else 256  # maximum episode length in training
    FOV_SIZE = 3    # 3 for SCRIMP, 11 for PRIMAL2
    WORLD_SIZE = (10, 40)
    OBSTACLE_PROB = (0.0, 0.5)
    ACTION_COST = -0.2 if LIFELONG else -0.3    # -0.3 for BOTH
    IDLE_COST = -0.2 if LIFELONG else -0.3      # -0.3 for BOTH
    GOAL_REWARD = 2.0 if LIFELONG else 0.0   # 0 for SCRIMP, +5 for PRIMAL2
    COLLISION_COST = -2    # -2 for BOTH 
    BLOCKING_COST = 0.0 if LIFELONG else -1      
    PRIMAL_MAZE = True
    MIN_DIST_NEW_GOAL = 5  # minimum distance between new goal and old goal


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    IN_VALUE_COEF = 0.08
    EX_VALUE_COEF = 0.08
    POLICY_COEF = 10
    VALID_COEF = 0.5
    BLOCK_COEF = 0.5
    N_EPOCHS = 10
    N_ENVS = 1  # number of processes 8 for training 1 for debug
    N_MAX_STEPS = 3e7  # maximum number of time steps used in training (1/2e7 for training,3e7 for debug)
    N_STEPS = 2 ** 10  # number of time steps per process per data collection (2**8 for training, 2**10 for debug)
    MINIBATCH_SIZE = int(2 ** 10)   # same as N_STEPS
    DEMONSTRATION_PROB = 0.1  # imitation learning rate


class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 8  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 7  # [dx, dy, d total,extrinsic rewards,intrinsic reward, min dist respect to buffer, action t-1]
    N_POSITION = 1024  # maximum number of unique ID
    D_MODEL = NET_SIZE  # for input and inner feature of attention
    D_HIDDEN = 1024  # for feed-forward network
    N_LAYERS = 1  # number of computation block
    N_HEAD = 8
    D_K = 32
    D_V = 32


class TieBreakingParameters:
    DIST_FACTOR = 0.1


class IntrinsicParameters:
    K = 3  # threshold for obtaining intrinsic reward
    CAPACITY = 80
    ADD_THRESHOLD = 3
    N_ADD_INTRINSIC = 1e6  # number of steps to start giving intrinsic reward
    SURROGATE1 = 0.2
    SURROGATE2 = 1


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = False
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = False
    WANDB =  False
    TENSORBOARD = False
    TXT_WRITER =  True
    MAKE_GIF = True
    ENTITY = 'jianxiongtho'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'MAPF'
    EXPERIMENT_NAME = 'SCRIMP2'
    EXPERIMENT_NOTE = 'LIFELONG' if EnvParameters.LIFELONG else 'ONESHOT'
    SAVE_INTERVAL = 5e5  # interval of saving model
    BEST_INTERVAL = 0  # interval of saving model with the best performance
    GIF_INTERVAL = 1e6  # interval of saving gif
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + ' ' + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + ' ' + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + ' ' + TIME
    TXT_NAME = 'alg.txt'
    LOSS_NAME = ['all_loss', 'policy_loss', 'policy_entropy', 'critic_loss_in', 'critic_loss_ex', 'valid_loss',
                 'blocking_loss', 'clipfrac',
                 'grad_norm', 'advantage']


all_args = {'N_AGENTS': EnvParameters.N_AGENTS, 'N_ACTIONS': EnvParameters.N_ACTIONS,
            'EPISODE_LEN': EnvParameters.EPISODE_LEN, 'FOV_SIZE': EnvParameters.FOV_SIZE,
            'WORLD_SIZE': EnvParameters.WORLD_SIZE,
            'OBSTACLE_PROB': EnvParameters.OBSTACLE_PROB,
            'ACTION_COST': EnvParameters.ACTION_COST,
            'IDLE_COST': EnvParameters.IDLE_COST, 'GOAL_REWARD': EnvParameters.GOAL_REWARD, 
            'COLLISION_COST': EnvParameters.COLLISION_COST,
            'BLOCKING_COST': EnvParameters.BLOCKING_COST,
            'LIFELONG': EnvParameters.LIFELONG,
            'ONLINE_FACTOR': EnvParameters.ONLINE_FACTOR,
            'MIN_DIST_NEW_GOAL': EnvParameters.MIN_DIST_NEW_GOAL,
            'lr': TrainingParameters.lr, 'GAMMA': TrainingParameters.GAMMA, 'LAM': TrainingParameters.LAM,
            'CLIPRANGE': TrainingParameters.CLIP_RANGE, 'MAX_GRAD_NORM': TrainingParameters.MAX_GRAD_NORM,
            'ENTROPY_COEF': TrainingParameters.ENTROPY_COEF,
            'IN_VALUE_COEF': TrainingParameters.IN_VALUE_COEF, 'EX_VALUE_COEF': TrainingParameters.EX_VALUE_COEF,
            'POLICY_COEF': TrainingParameters.POLICY_COEF,
            'VALID_COEF': TrainingParameters.VALID_COEF, 'BLOCK_COEF': TrainingParameters.BLOCK_COEF,
            'N_EPOCHS': TrainingParameters.N_EPOCHS, 'N_ENVS': TrainingParameters.N_ENVS,
            'N_MAX_STEPS': TrainingParameters.N_MAX_STEPS,
            'N_STEPS': TrainingParameters.N_STEPS, 'MINIBATCH_SIZE': TrainingParameters.MINIBATCH_SIZE,
            'DEMONSTRATION_PROB': TrainingParameters.DEMONSTRATION_PROB,
            'NET_SIZE': NetParameters.NET_SIZE, 'NUM_CHANNEL': NetParameters.NUM_CHANNEL,
            'GOAL_REPR_SIZE': NetParameters.GOAL_REPR_SIZE, 'VECTOR_LEN': NetParameters.VECTOR_LEN,
            'N_POSITION': NetParameters.N_POSITION,
            'D_MODEL': NetParameters.D_MODEL, 'D_HIDDEN': NetParameters.D_HIDDEN, 'N_LAYERS': NetParameters.N_LAYERS,
            'N_HEAD': NetParameters.N_HEAD, 'D_K': NetParameters.D_K, 'D_V': NetParameters.D_V,
            'DIST_FACTOR': TieBreakingParameters.DIST_FACTOR, 'K': IntrinsicParameters.K,
            'CAPACITY': IntrinsicParameters.CAPACITY, 'ADD_THRESHOLD': IntrinsicParameters.ADD_THRESHOLD,
            'N_ADD_INTRINSIC': IntrinsicParameters.N_ADD_INTRINSIC,
            'SURROGATE1': IntrinsicParameters.SURROGATE1, 'SURROGATE2': IntrinsicParameters.SURROGATE2,
            'SEED': SetupParameters.SEED, 'USE_GPU_LOCAL': SetupParameters.USE_GPU_LOCAL,
            'USE_GPU_GLOBAL': SetupParameters.USE_GPU_GLOBAL,
            'NUM_GPU': SetupParameters.NUM_GPU, 'RETRAIN': RecordingParameters.RETRAIN,
            'WANDB': RecordingParameters.WANDB,
            'TENSORBOARD': RecordingParameters.TENSORBOARD, 'TXT_WRITER': RecordingParameters.TXT_WRITER,
            'ENTITY': RecordingParameters.ENTITY,
            'TIME': RecordingParameters.TIME, 'EXPERIMENT_PROJECT': RecordingParameters.EXPERIMENT_PROJECT,
            'EXPERIMENT_NAME': RecordingParameters.EXPERIMENT_NAME,
            'EXPERIMENT_NOTE': RecordingParameters.EXPERIMENT_NOTE,
            'SAVE_INTERVAL': RecordingParameters.SAVE_INTERVAL, "BEST_INTERVAL": RecordingParameters.BEST_INTERVAL,
            'GIF_INTERVAL': RecordingParameters.GIF_INTERVAL, 'EVAL_INTERVAL': RecordingParameters.EVAL_INTERVAL,
            'EVAL_EPISODES': RecordingParameters.EVAL_EPISODES, 'RECORD_BEST': RecordingParameters.RECORD_BEST,
            'MODEL_PATH': RecordingParameters.MODEL_PATH, 'GIFS_PATH': RecordingParameters.GIFS_PATH,
            'SUMMARY_PATH': RecordingParameters.SUMMARY_PATH,
            'TXT_NAME': RecordingParameters.TXT_NAME}
