from .dataset import build_multithumos, build_charades, build_thumos14


def build_dataset(args):
    if args.dataset == 'multithumos':
        return build_multithumos(args)
    elif args.dataset == 'charades':
        return build_charades(args)
    elif args.dataset == 'thumos14':
        return build_thumos14(args)
    elif args.dataset == 'activitynet':
        return build_activitynet(args)
    else:
        raise ValueError(f'dataset {args.dataset} not implemented')
