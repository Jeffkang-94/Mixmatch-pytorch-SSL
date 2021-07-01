
def train_one_epoch(epoch,
                    model,
                    criterion,
                    optim,
                    lr_schdlr,
                    ema,
                    dltrain_x,
                    dltrain_u,
                    lambda_u,
                    n_iters,
                    logger,
                    log_interval,
                    alpha,
                    T,
                    n_class
                    ):

    model.train()

    loss_meter = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()

    epoch_start = time.time()  # start time
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters):
        ims_x, _, targets_x = next(dl_x)
        ims_u1, ims_u2, _ = next(dl_u)

        bt = ims_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(bt, n_class).scatter_(1, targets_x.view(-1,1).long(), 1)

        ims_x, targets_x = ims_x.cuda(), targets_x.cuda()
        ims_u1, ims_u2 = ims_u1.cuda(), ims_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u1, _ = model(ims_u1)
            outputs_u2, _ = model(ims_u2)
            p = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([ims_x, ims_u1, ims_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        lamda = np.random.beta(alpha, alpha)
        lamda = max(lamda, 1-lamda)

        newidx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[newidx]
        target_a, target_b = all_targets, all_targets[newidx]

        mixed_input = lamda * input_a + (1 - lamda) * input_b
        mixed_target = lamda * target_a + (1 - lamda) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, bt))
        mixed_input = mixmatch_interleave(mixed_input, bt)

        logit, _ = model(mixed_input[0])
        logits = [logit]
        for input in mixed_input[1:]:
            logit, _ = model(input)
            logits.append(logit)

         # put interleaved samples back
        logits = mixmatch_interleave(logits, bt)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        loss_x, loss_u, w = criterion(logits_x, mixed_target[:bt], logits_u, mixed_target[bt:], epoch+it/n_iters, lambda_u)

        loss = loss_x + w * loss_u

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        loss_meter.update(loss.item())
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())

    ema.update_buffer()

 
    return loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg