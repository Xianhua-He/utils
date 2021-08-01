def _init_weight(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


if args.pretrain:
    if os.path.isfile(args.pretrain):
        if dist.get_rank() == 0:
            print("=> loading pretrained weights from '{}'".format(args.pretrain))
        checkpoint = torch.load(args.pretrain, map_location="cpu")
        for k in list(checkpoint.keys()):
            if not k.startswith('fc.'):
                checkpoint['encoder.' + k] = checkpoint[k]
                del checkpoint[k]
        del checkpoint["fc.weight"]
        del checkpoint["fc.bias"]
        msg = model.load_state_dict(checkpoint, strict=False)
        if dist.get_rank() == 0:
            print(msg.missing_keys)
    else:
        if dist.get_rank() == 0:
            print("=> no pretrained weights found at '{}'".format(args.pretrain))


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume, map_location="cpu")
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))