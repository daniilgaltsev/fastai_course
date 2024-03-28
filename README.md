# Fast.ai Course

## TODO

- write what was done generally
- deal with hws, selfs and todos
- try improving / rerunning all models to get better performance
- edit the notes to be more readable


## Setup
- I used WSL2 on Windows 10.
- for managing the environment I used mamba.
- createed the enviroment with python\=\=3.11.
- installed jupyterlab and nbdev
- ran `nbdev_install_hooks`
- ran `nbdev_new --lib_name miniai  --lib_path miniai `
- ran `pip install -e .`
- removed .github/, LICENSE
- added to .gitignore: \_proc, \_docs, .gitattributes, egg



---

## Lesson 11 (second half)

- add missing requirements

### 01_matmul

#### get data
- get mnist data
- read it
- display an example
- use itertools to get an image
###### notes
- uhhhh, what's Not a gzipped file (b'\n\n')
- oh, i've messed up the download link
- oh, so iter can also accept a sentinel
#### matrix and tensor
- write matrix class
	- init, get
- use map to transform data to tensors
- reshape train
- display an example
#### random numbers
- write random (rnd_state, seed(a)) based on Wichmann Hill
- write rand()
- show that does not work properly when forking
- show the same behaviour in torch
- display plot and hist for our random
- time using our random and torch random (aka weight initialization from x to labels)
###### notes
- ehhhhh, how do we seed it?
- also don't forget to exit in children when forking
#### matrix multiplication
- fix torch seed
- generate weights and bias
- make a small subset to iterate with
- write python matrix multiplication
- can set printoptions for pretty print (torch and numpy)
- write matmul(a,b)
- time it
#### numba
- write jitted dot(a,b)
- time it x2 to show jitting
- rewrite matmul(a,b) to remove 1 loop
- test that results are close
- and time it to compare
###### notes
- if you use just list in numba it says it's deprecated so use np arrays
- huh?!: ModuleNotFoundError: No module named 'fastcore'
- reinstalled the environment and it works now (probably messed up the base environment)
- oh, numba does not work with torch
#### elementwise ops
- play with elementwise (add, mean, frobenius norm, sqrt)
- rewrite matmul(a,b) with 2 loops using only torch
- test that results are close
- and time it to compare
#### broadcasting
##### broadcasting with a scalar
- play with broadcasting with a scalar (less, add, multiply)
##### broadcasting a vector to a matrix
- play with broadcasting a vector to a matrix (m + v, v +m)
- use .expand_as to explain how that works
- look at .storage
- look at .stride
- play with None, unsqueeze, ...,  to replicate broadcasting
##### broadcasting rules
- recall the broadcasting rules
- write examples to show them
#### matmul with broadcasting
- show how .expand_as works
- write matmul(a, b) with 1 loop
- test that close
- time it to compare
- now time with full data



---

## Lesson 12


### 01_matmul

#### einstein summation
- recall the einsum rules
- write matmul(a,b) with einsum
- test it
- time it
#### pytorch op
- test torch @
- time torch.matmul
#### cuda
- write matmul(grid, a, b, c) as if a gpu kernel (fills one element)
- show that fills one element in result
- write launch_kernel(kernel, grid_x, grid_y, \*args, \*\*kwargs)
- write @cuda.jit matmul(a, b, c)
- calculate TPB, blockspergrid
- show that works
- test it
- time it
- now do multiplication with torch.cuda
- time it
###### notes
- oh, forgot to install cudatoolkit


### 02_meanshift

#### clustering
- set seed and printoptions
##### create data
- set n_clusters and n_samples
- generate centroids
- use torch.distributions to generate samples (sample(m))
- plot the data
##### mean shift
- write gaussian(d, bw)
- write plot_func(f) to plot a function
- show gaussian plot
- write tri(d, i)
- show tri plot
- clone data
- choose a random point
- calculate Eucledian distances for it
- HW: rewrite using einsum
- calculate weights
- calculate weighted average
- write one_update(X) (with for over items)
- write meanshift(data)
- time it
- plot calculated centroids with true centroids
- run with tri
###### notes
- is there a practical difference in updating one at a time vs. updating all at once
- ok, the difference might be when cluster overlap
- in this case it doesn't seem to be the case but it might actually be true for slightly overlapping clusters depending on how we iterate through points
- I had a bug with incorrect names -> tri does not fully converge with bandwidth
##### animation
- write do_one(d)
- create animation using matplotlib.animation and IPython.display
###### notes
- ok, we need to pass ax to plot_clusters
##### gpu batched algorithm
- sample a batch for iterating
- write dist_b(a, b)
- calculate dist for the batch
- calculate weights for the batch
- calculate new points for the batch
- write meanshift(data, bs=500)
- time it for different bs
- plot result
###### notes
- somehow wrote it to be transposed
- my version is not that fast for larger batch sizes
- ok, i guess using the matmul directly is much much better
- no, it actually was my way of calculating distances


### Homework
- Implement k-means clustering, dbscan, locality sensitive hashing, or some other clustering, fast nearest neighbors, or similar algorithm of your choice, on the GPU. Check if your version if faster than a pure python or CPU version. TODO
- Try to create animation for it (or for stable diffusion) TODO
- Write a meanshift that picks only the closest points to avoid quadratic time TODO
###### notes
- need to create more points



---

## Lesson 13


### 03_backprop

#### the forward and backward passes
- imports, seed, printoptions, data loading (train, valid as tensors)
#### foundations version
##### basic arcitecture
- generate two sets of weights (defined by n_hidden and only 1 output)
- write lin(x, w, b) (linear)
- run linear
- write relu(x)
- run relu
- write model(xb) (L->relu->L)
- run model
##### loss function: MSE
- write mse(output, target)
- run mse
##### gradients and backward pass
- use sympy to calculate some derivatives
- write lin_grad(inp, out, w, b)
- write forward_and_backward(inp, targ)
- run it
- write get_grad(x)
- save calculated grads
- use pytorch autograd to
	- write mkgrad(x) (generate new tensors)
	- write forward(inp, targ)
	- calculate loss
	- calculate grads
	- test against previous
###### notes
- the gradients are incorrect - gonna check it by doing the checks in reverse order
- ok, forgot to sum over and  confused a pair of args
#### refactor model
##### layers as classes
- write Relu():
	- call(self, inp)
	- backward(self)
- write Lin():
	- init(self, w, b)
	- call(self, inp)
	- backward(self)
- write Mse():
	- call(self, inp, targ)
	- backward(self)
- write Model():
	- init(self, w1, b1, w2, b2)
	- call(self, x, targ)
	- backward(self)
- run model
- test against previous gradients
##### Module.forward()
- write Module():
	- call(self, \*args)
	- forward(self)
	- backward(self)
- rewrite Relu, Lin, Mse
- run model and test against previous gradients
##### Autograd
- rewrite Linear, Model using nn.Module (but generate weights)
- run model
- look at gradients
###### notes
- oh, yeah, since we are using mse targets need to be float


### 04_minibatch_training

- same first cell as 03 (imports, etc.)
#### initial setup
##### data
- same model except output is all classes
- and run it
##### cross entropy loss
- write the not good log_softmax(x)
- run it on predictions
- write better log_softmax(x)
- write logsumexp(x)
- test against torch's
- write better log_softmax(x)
- write nll(input, target)
- test against torch's
#### basic training loop
- run model on a batch
- calculate loss
- get predicted classes
- write accuracy(out, yb)
- write training loop (define lr and epochs)
	- print out loss and accuracy



---

## Lesson 14

- add datasets, torchvision to requirements.txt


### 04_minibatch_training

#### basic training loop
- rewrite logging with a report(loss, preds, yb) and f-strings
#### using parameters and optim
##### parameters
- show that nn.Module can track assigned Modules
- named_children
- parameters
- write MLP(nn.Module)
- create model
- show all stuff in model
- write fit()
- write MyModule() that also can track assigned Modules and show them
##### registering modules
- write Model(nn.Module)
	- init(self, layers)
##### nn.ModuleList
- write SequentialModel using ModuleList
- fit it
##### nn.Sequential
- create model and fit using Sequential
##### optim
- write Optimizer():
	- init, step, zero_grad
- fit model using that
- write get_model (uses torch's)
- fit model using that
###### notes
- i always forget to disable grad - at least when updating weights it breaks
#### dataset and dataloader
##### dataset
- write Dataset()
- create ds_train, ds_valid
- get a train batch
- check shapes
- fit model using that
##### dataloader
- write DataLoader()
- create dl_train, dl_valid
- get a train batch
- write fit() with that
##### random sampling
- write Sampler()
- show how it outputs indices
- write BatchSampler() (here using fastcore's store_attr can be useful)
- get a train batch
- write collate(b)
- rewrite DataLoader()
- create samplers, dataloaders
- get a train batch
- fit model
##### multiprocessing dataloader
- rewrite DataLoader using a Pool
- create dataloader and get a batch
###### notes
- for some reason it hangs
- ok, it hangs with the BatchSampler
- ok, it just hangs with 4 items
- well, it also depends on the size of the tensor
- ok, it doesn't break when i run through a script
- i don't want to deal with it, so, i'll just ignore it and hope it does not cause problems later
- oh, so, now that I decided to ignore it, it finally gave me an error ('RuntimeError('unable to open shared memory object <\/torch_50829_3284495858_2010> in read-write mode: Too many open files (24)')') - ok, that's something, but I'm still ignoring it - i copied the code from original nb and it still hangs - running original in colab works, mine too works in colab - i guess it's a problem with wsl and/or how i setup the environemnt

##### pytorch dataloader
- create all of these using torch's 
- fit model
- show the torch's shortcuts
- show we can do the same without batch_sampler and collate
###### notes
- there are both sampler and batch_sampler in torch's dataloader
- ok, it seems that torch works fine - probably messed up somewhere than, but i don't want to look for it
#### validation
- rewrite fit to also do validation
- write get_dls
- write the 3 lines to train (time fit)


### 05_datasets

- do initial setup (copy from the nb)
#### hugging face datasets
- load fashion_mnist dataset builder
- look at descriptions, features, splits
- load the dataset
- get train, test
- look at one example
- get a batch
- get names for classes for the batch
- write collate_fn(b)
- create dataloader
- get batch from it
- write transforms(b)
- use datasets's `thingy` with transforms
- create dataloader from the `thingy` and get a batch
- write \_transfromi(b) (inplace)
- write inplace(f)
- use it to get transformi (that returns)
- write the previous thing replacing trransforms with transformi
- show the decorator syntax
- show how to use itemgetter (with dict and a custom class)
- show default_collate with dicts
- write collate_dict(ds) that return \_f(b)
- now use it to get a dataloader and a batch
- make this notebook default_exp datasets
- export collate_dict, inplace and necessary imports
- add nbdev.nbdev_export at the end of the notebook
#### plotting images
- show an example image from a batch
- write and export show_image(im, ax, figsize, title, \*\*kwargs) (also use fc.delegates to imshow)
- show help of it
- show an image using it
- show 2 images on one plot
- write and export subplots(nrows, ncols, figsize, imsize, suptitle, \*\*kwargs) with full help
- use it to plot 8 images in 3x3
- write and export get_grid(n, nrows, ncols, title, weight, size, \*\*kwargs) with full help
- use it to plot 8 images 3x3 without the bad things at the end
- write and export show_images(ims, nrows, ncols, titles, \*\*kwargs) with full help
- use it to show 8 images with their classes
###### notes
- i somehow messed up the figure when it's 2x4 or 4x2 (at least) - i mixed up nrows and ncols when setting figsize
- oh, that's why they had such a weird operation with titles - they don't fit
#### export
- already should be filled in


### 06_foundations

- copy imports
#### callbacks
##### callbacks and gui events
- use ipywidgets.widgets.Button to make a button that writes `hi`
###### notes
- can't get the output - oh, my log window was just completely minimized
##### creating your own callback
- write slow_calculation
- call it
- write slow_calculation(cb)
- write show_progress(epoch)
- call slow_calculation so that it prints epoch
##### lambdas and partials
- write show_progress(exclamation, epoch)
- call slow_calucation with it
- write make_show_progress(exclamation)
- call slow_calculation with it
- do the same with partial
##### callbacks and callable classes
- write ProgressShowingCallback()
- call slow_calculation with it
##### multiple callback funcs; `*args` and `**kwargs`
- rewrite slow_calculation(cb) to pass result to cb with before_calc and after_calc
- write PrintStepCallback()
- use it with slow_calculation
- rewrite PrintStepCallback() that receives named args
- use it with slow_calculation
###### modifying behaviour
##### `__dunder__` things
- write SloppyAdder (add with 0.01)
- show how it works
###### `__getattr__` and `getattr`
- write A() with 2 properties
- show getattr with it
- write B() with custom getattr
- show it



---

## Lesson 15

- add fastprogress to requirements.txt
- make sure that correclty exports datasets from 05_datasets
- make sure that correctly exports training from 04_minibatch_training (accuracy, report, dataset, pytorch: DataLoader, SequentialSampler, RandomSampler, BatchSampler; fit, get_dls)


### 07_convolutions

#### convolutions
- set this notebook as conv export
- copy imports and data load
##### understanding the convolution equations
- explain kernel and how result is produced
- display an image
- write top_edge kernel
- show it
- write apply_kernel(row, col, kernel) for the image
- apply to image and show result
- do the same for left_edge
###### notes
- oh, so just adding ; to the end will stop jupyter from displaying the last thing
##### convolutions in pytorch
- explain im2col
- (optional hw) write it
- use torch's version to apply a convolution
- show the result
- timeit the 2 versions and conv2d
- write 2 diag kernels
- get a batch of images and batch the kernels
- apply them batched
- show the results for an image
###### notes
- can probably numpify (numpyify?) im2col_conv more, but it's good enough for understanding what needs to be done for it to work - ok, first, I'll try to deal with the last kernel loop since we can probably unroll it with a reshape - yeah, that worked, now deal with inner 2 loops with better indexing -  the loops copies into A\[r\] a block img\[:, i:i+k, j:j+k\], maybe there is a simple order - oh, yeah, it can just be flattened - not sure that the last 2 loops can be removed because of the overlapping windows, but maybe there is a numpy function that i don't know aobut
- why am i getting the same result for the 2 diags? - i messed up names and didn't restart and rerun all
##### strides and padding
- explain padding
- explain stride
#### creating the cnn
- create a _broken_ cnn (1->30->10)
- show the output shape
- write and export conv(ni, nf, ks=3, stride=2, act=True) + padding
- create a simple_cnn
- show the output shape
- reshape images
- create datasets with them
- write def_device, to_device(x, device), collate_device(b) and export them
- prepare dataloaders and sgd optimizer
- fit the model
- try to reach similar accuracy by playing with lr (train with reduced lr, 2 times, etc.)
- compare the number of parameters between the models 
###### notes
- that's why we imported Mapping
- it was slow, but I just completely forgot to move to the correct device both the data and the model
##### understanding convolution arithmetic
- explain shape of a batch of images
- explain receptive field


### 08_autoencoder

#### autoencoders
- copy imports
##### data
- load fashion_mnist
- write transformi(b)
- transform images and show an example
- collate the data
- write collate_(b) and data_loaders(dsd, bs, \*\*kwargs)
- create dataloaders (collate_, data_loaders)
- get a batch
- show 16 images with classes
###### notes
- for datasets it's much better (faster) to first access the item by index and then features
##### warmup - classify
- fit the same model from the previous nb
- why does it fit so slowly
- increase the number of workers
- why doesn't it work
##### autoencoder
- write deconv(ni, nf, ks, act) using nn upsampling+conv
- write eval(model, loss_func, valid_dl, epoch)
- write fit(epochs, model, loss_func, opt, train_dl, valid_dl)
- create autoencoder with 3 convs
- evaluate it
- fit it with mse (it is fiddly, might need to play around with reducing lr, optimizers)
- show results vs originals
###### notes
- the shape is different so need to remove padding at the end, and, probably, add it in the beginning for symmetry


### 09_learner

- export it as learner
- copy imports
- export imports
#### learner
- copy loading fashion_mnist from previous nb
- write and export DataLoaders:
	- init(self, \*dls)
	- from_dd(cls, dd, batch_size, as_tuple, num_workers)
- create dataloaders and get a batch
- write Learner: (saving a lot of stuff as properties)
	- init(self, model, dls, loss_func, lr, opt_func)
	- one_batch(self)
	- calc_stats(self)
	- one_epoch(self, train)
	- fit(self, n_epochs)
- create a model with 2 Linear
- create learner
- fit it
###### notes
- what's as_tuple?
- i don't really like implicity passing arguments through object properties, but I guess I'll try it here
- forgot to flatten the images in transformi
- i should also add accuracy
#### metrics
- write and export Metric:
	- init(self)
	- reset(self)
	- add(self, inp, targ, n)
	- value(self)
	- calc(self, inps, targs)
- write and export Accuracy(Metric)
- show how Accuracy works
- show how to use Metric for loss
###### notes
- for Metric to work nicely with loss, need to add some default values
- ok, it's a bit trick to correctly calculate means for different metrics
#### callbacks learner
- write and export identity(\*args)
- write and export CancelFitException, CancelBatchException, CancelEpochException
- write and export with_cbs: (before_\<name\>, after_\<name\>)
	- init(self, nm)
	- call(self, f) (\_f(o, \*args, \*\*kwargs), catch exceptions from above by name)
- write and export Learner: (store learn as property,  use private for with_cbs and main for setup, maybe better to write a fit first so that can iterate faster)
	- init(self, model, dls, loss_func, lr, cbs, opt_func)
	- one_batch(self) (look at getattr)
	- one_epoch(self, train)
	- \_one_epoch(self)
	- fit(self, n_epochs)
	- \_fit(self)
	- getattr(self, name) (predict, get_loss, backward, step, zero_grad; self.callback) 
	- callback(self, method_nm)
- write and export Callback() (it stores order)
- write and export TrainCB(Callback):
	- predict(self)
	- get_loss(self)
	- backward(self)
	- step(self)
	- zero_grad(self)
- write and export DeviceCB(Callback):
	- before_fit(self)
	- before_batch(self)
- write and export MetricsCB(Callback): (both loss and an additional metric; also add self to learn)
	- init(self, metric)
	- log(self, \*s)
	- before_fit(self)
	- before_epoch(self)
	- after_epoch(self)
	- after_batch(self)
- write and export ProgressCB(Callback): (using fastprogress, order)
	- init(self, plot=False)
	- before_fit(self) (overwrite learn.epochs, overwrite learn.metrics.log)
	- log(self, \*s) (write to the bar)
	- before_epoch(self) (overwrite learn.dl)
	- after_batch(self) (update dl.comment, and plot)
- create a model
- create the learner with cbs
- fit it (it should report loss, accuracy and plot a graph)
###### notes
- oh, yeah, in cbs it's quite annoying to be constantly adding .learn. should probably add a getattr to do that, but i think that's done later in the course anyway, so i'll suffer for now
- for some reason fc.store_attr breaks - i forgot to return an error from getattr
- was confused why m.reset() was breaking -> forgot to initialize Accuracy vs. Accuracy()



---

## Lesson 16

- add torcheval to requirements.txt


### 09_learner

- goes through basic callbacks learner and then flexible learner which is actually a more gradual way to build the learner in the previous lesson
#### metrics
- export torcheval's MulticlassAccuracy, Mean
- play with them
- stop exporting our own Metric, Accuracy
	- write and export to_cpu(x)
- write and export new MetricsCB(Callback):
	- similar to old but uses torcheval's metrics (use to_cpu since at that time it broke sometimes)
	- support multiple metrics
	- add automatic metric name logging
- fit model with it
#### flexible learner
- rewrite Learner to also use callback_ctx(self, nw) with context manager (actually don't because they had problem with exception and yield, continue using decorator with_cbs)
#### traininglearner subclass
- write and export MomentumLearner(Learner):
	- predict, get_loss, backward, step, zero_grad
- fit a learner without TrainCB
#### LRFinderCB
- write LRFinderCB(Callback) (stop when loss is 3 times larger than min)
		- init, before_fit, after_batch, after_fit
- run it
- plot the results
- export torch's ExponentialLR
- rewrite and export LRFinderCB(Callback) using ExponentialLR
- run it and plot the results
###### notes
- ok, something weird is going with lrfindercb - gets to very high learning rate 10^6 and only then loss increases from ~2 to a lot - first, need to cancel batch if not training - and that was it, lol
- also, doesn't call after_fit because of the raised exception, but I think they change it later and i'll do the same
- better to check with the nb in repo to check that everything (that is exported) is ok
- checked and:
	- add run_cbs
	- add SingleBatchCB
	- in MetricsCB update log to just print and add epoch,train to dict (i think that so it works better with ProgressCB)
	- also update all callbacks to accept learn
	- add device to DeviceCB init, also check if model has .to
	- add n_inp to TrainCB init
	- add self.first to ProgressCB to make a table, add valid loss (it's actually not easy to correctly plot validation loss with train loss)
	- forgot order in progresscb (it's not critical but still)
	- add cleanup callback to with_cbs
	- lots of updates to Learner to make it easier to use (callbacks, default, etc.)
	- add TrainLearner
	- update MomentumLearner to use TrainLearner
	- add lr_find to Learner using patch
	- move loss to cpu and also check loss for nan
- after all that still:
	- does not show plot after lr find (makes sense since we didn't do anything with that and I probably forgot to move it to the finally callback - yes, i did)
	- also does not plot progress because there is no val_loss - once again, forgot to add it after epoch - was a bit confused how it was calculated, but, yeah, the metrics are reset before each epoch
	- also, switched dls.train and dls.valid in Learner
- additionally, added update_rate to ProgressCB


### 10_activations

- export as activations
#### activation stats
- copy and export imports
- write and export set_seed(seed)
- load fashion_mnist
##### baseline
- write fit(model, epochs)
- fit the same CNN as before (use high lr so it breaks and use seed)
###### notes
- i am gonna copy the batch size
- for some i had to use much higher lr to break it, though it was definitely bad before but not obvious obvious bad
##### hooks 
###### manual insertion
- write SequentialModel with manual storage of means and stds of weights per layer
	- init, call, iter
- train it
- plot activation means
- plot activation stds
###### pytorch hooks
- create act_means, act_stds for holding
- write append_stats(i, mod, inp, outp) hook
- register the hook
- fit the model
- plot the results
###### hook class
- write and export Hook():
	- init(self, m, f)
	- remove(self)
	- del(self)
- write append_stats(hook, mod, inp, outp)
- register hook, fit model
- plot the results
###### a hooks class
- write and export Hooks(list):
	- init(self, ms, f)
	- enter(self, \*args)
	- exit(self, \*args)
	- del(self)
	- delitem(self, i)
	- remove(self)
- fit model with Hooks and plot the results
##### histograms
- write and export append_stats(hook, mod, inp, outp) (it adds histogram of absolute values)
- train model
- plot the results
	- for each layer plot a histogram (log1p)
	- same as the other but only looking at the 2 smallest group proportion to all
- write and export get_hist(h)
- write and export get_min(h)
###### notes
- lol, how do i flip it? - you can't reverse using slice - oh, there is an origin arg in plot - why does it not work? - I forgot to pass kwargs...



---

## Lesson 17


### 09_learner

- rewrite Callback to
	- add getattr for model, opt, batch, epoch from learn
	- add training
- write and export TrainLearner(Learner) to replace TrainCB
	- also need to chnage MomentumLearner
###### notes
- wait, learn is now passed directly, so, don't do that?
- also the others were already done when checking during the previous lesson


### 10_activations

- write and export HooksCallback(Callback): 
	- init(self, hookfunc, mod_filter=fc.noop)
	- before_fit, after_fit, iter, len
	- \_hookfunc(self, \*args, \*\*kwargs)
- rewrite fit(model, epochs, xtra_cbs)
- create it with append_stats
- fit the model
- plot the results
- write and export ActivationStats(HooksCallback):
	- init, color_dim, dead_chart, plot_stats, 
- rewrite get_min(h) to use only the min bin instead of 2
- create the callback
- fit model with it
- show the 3 charts
###### notes
- why do we need \_hookfunc? - looked at their code and it's for running hookfunc during correct stages (train/val)


### 11_initializing

- export as init
#### Initialization
- copy and export imports
- load fashion_mnist
- write get_model() 1->8->16->32->64->10
- train model with cbs and momentum (it might break with LRFinder)
- try to get a good accuracy and plot the diagnostics to see that it's not working
- copy and export the 3 functions to clean outputs to free memory
###### notes
- i guess to save memory, we should call lr_find inline without assigning variables
- the model does recover, but there is definitely a nasty spike which prevents us from training faster and with a higher learning rate
##### glorot/xavier init
- show my randn init will blow up or become 0 with a lot of layers
- now show that xavier init works
###### background
###### variance and standard derivation
###### covariance
###### xavier init derivation
##### kaiming/he init
###### background
- show that xavier has problems when we add relu
- show that he fixes it
##### applying an init function
- print all module names using apply
- write init_weights(m) using init.kaiming_normal_
- use it on the model
- fit the model
- look at diagnostics
###### notes
- why did the dataloader break in transformi with key error? - oh, I renamed x. yeah, that would do that
##### input normalization
- get mean and std of data
- write and export BatchTransformCB(Callback):
	- init(self, tfm)
	- before_batch(self)
- write \_norm(b)
- fit the model with the transform and look at diagnostics
- write transformi(b) to pretransform data
#### general relu
- write and export GeneralRelu(nn.Module)
	- init(self, leak, sub, maxv)
	- forward(self, x)
- write and export plot_func(f, start, end, steps)
- plot some variations of GeneralRelu
- write conv(ni, nf, ks, stride, act)
- write get_model(act, nfs)
- write and export init_weights(m, leaky)
- create CBs and model
- fit the model and plot diagnostics
#### lsuv
- write and export \_lsuv_stats(hook, mod, inp, outp)
- write and export lsuv_init(model, m, m_in, xb) (don't forget to check if explodes to not hang)
- get layers to init
- initialize each layer
- fit the model and plot diagnostics
###### notes
- ok, it breaks for me even though the initialization seems good - i for some reason reversed the order which is not right for initialization but still my mean and std aren't converging - ok, i've just got completely confused with args
#### batch normalization
##### layernorm
- write LayerNorm(nn.Module)
	- init(self, dummy, eps), forward
- write and export conv(ni, nf, ks, stride, act, norm, bias) (also immediately correctly handle bias in case of torch's batchnorm)
- write and export get_model(act, nfs, norm)
- fit the model and plot diagnostics
###### notes
- why is there a dummy? - ah, it's for when we need the number of channels in batchnorm
- ok, i've also noticed that i've messed up init_weights before (i passed leak to m instead of leak) but how did it still work, lol - now, it trains better
##### batchnorm
- write BatchNorm(nn.Module) (self.register_buffer)
	- init(self, nf, mom, eps), update_stats, forward
- fit the model and plot diagnostics
###### notes
- forgot to turn of stats updating during validation and grads in general
#### towards 90%
- do the final fit (maybe decrease bs and also train with decreasing lr; also start using torch's versions)


### 12_accel_sgd

- export it as sgd
- copy and export imports
- load fashion_mnist
- prepare cbs, activation, init_weights
#### sgd
- write SGD:
	- init(self, params, lr, wd)
	- step(self)
	- opt_step(self, p)
	- reg_step(self, p)
	- zero_grad(self)
- fit model
- plot color dim
###### notes
- it seems like it stops training after the first batch - oh, we need to convert params to a list and not leave it as a generator
##### momentum
- write Momentum(SGD)
- fit model
- plot color dim
###### notes
- had to use a larger lr to see an effect
#### rmsprop
- write RMSProp(SGD):
- fit model
- plot color dim
###### notes
- here, I need to decrease lr to make it work
- ok, yeah, initializing square averages to zero really works worse
- also, what if we just don't average it out and use the latest value with epsilon clipping?
#### adam
- write Adam(SGD)
- fit model
- plot color dim
###### notes
- similar to how was done by JH in RMSProp - why not initialize to not zeros - oh, wait, since first step is 0 (+1), it becomes the full value at first


### Homework

- write LsuvCB



---

## Lesson 18

- (hw) create an automated annealer
- rewrite Learner's fit(self, n_epochs, train, valid, cbs, lr)
- write patch lr_find(self:Learner, gamma, max_mult, start_lr, max_epochs)


### 12_accel_sgd

#### schedulers
- create a learner and fit it for one batch
- get an optimizer from it
- get all parameters
- get the state of optimizer
- look at them
- look at param groups
- create a CosineAnnealingLR
- look at it
- plot how it works
##### scheduler callbacks
- write and export BaseSchedCB(Callback):
	- init(self, sched), before_fit, \_step
- write and export BatchSchedCB(BaseSchedCB)
- write and export HasLearnCB(Callback)
	- modify run_cbs, Learner and callback methods to also pass learn
- write and export RecorderCB(Callback)
	- init(self, \*\*d), before_fit, after_batch, plot
- write \_lr(cb)
- setup scheduler and cbs
- fit model
- plot the Recorded
- write and export EpochSchedCB(BaseSchedCB)
- fit model and plot the Recorded
###### notes
- forgot to switch from batch to epoch cb, but it still trained well (0.875 vs 0.863 vs 0.876)
- need to name it \_step in sched callback so that it does not overwrite the main step
##### 1cycle training
- write \_beta1(cb)
- fit with OneCycleLR (maybe train for longer and play with lr to get that it over 0.9)
- plot lr and momentum
###### notes
- barely over the line, lol


### 13_resnet

#### resnets
- export as resnet
- copy and export imports
- load fashion_mnist
- create and export act_gr (from GeneralRelu)
- create cbs and init_weight
##### going deeper
- write get_model(act, nfs, norm) (bigger CNN than before)
- fit the model
##### skip connections
- write and export \_conv_block(ni, nf, stride, act, norm, ks)
- write and export ResBlock(nn.Module)
	- init(ni, nf, stride, ks, act, norm)
	- forward ( act( convs(x) + idconv(pool(x)) ) )
- write get_model(act, nfs, norm) (using resblocks)
- write \_print_shape(hook, mod, inp, outp)
- create model and print shapes
- patch summary(self:Learner) (|Module|Input|Output|Num params| + total params)
	- also can create markdown table using fastcore.IN_NOTEBOOK and IPython
- run summary on a model
- fit model
###### notes
- the sizes didn't align until I changed avg pool to also use ceil mode
- also, now cublas breaks because some operation does not support deterministic, it says to set an env variable, but I guess I'll just add an option to set seed to not be deterministic


### 14_augment

- export as augment
#### augmentation
- copy and export imports
- load fashion_mnist
- prepare cbs and init_weights
##### going wider
- write get_model(act, nfs, norm) (more filters + initial conv with bigger kernel)
- fit model
###### notes
- oh, i've already used a larger kernel for the first layer
- also, now 0.2 is too large
###### pooling
- write GlobalAvgPool(nn.Module)
- write get_model2 (diff in last layer)
- write and export \_flops(x, h, w)
- rewrite and export patch summary to include ~MFLOPS + total MFLOPS
- run summary on model
- fit model
- write get_model3 to reduce num params and MFLOPS by removing a block
- fit model
- write get_model4 reduce MFLOPS in first block
- fit model
##### data augmentation
- train for 20 epochs (should see 99.9% train and the same as before val)
- write tfm_batch(b, tfm_x, tfm_y)
- create a transforms model (crop and flip)
- create batchtransformcb (also need to rewrite to also have on_train and on_valid)
- get and fit model with SingleBatchCB
- get the batch and show it
- now fit for 20 epochs with augmentations
- (HW) try to beat 93.8 (w TTA 94.2) on val in 20 epochs
- save the model (torch.save)
###### notes
- also, if weight decay has less of an impact because we have only 'one' scale parameter, why not reweight to have a bigger penalty? - doesn't seem to have a substantial effect? if anything it might be better to not use wd at all
- oh, i didn't use the largest model - that would explain the lower score
- still not training well + training accuracy is low - maybe decrease the crop so that it's 'easier' and reduce lr by a bit since there is small spike in the loss graph (for both)
- let's try adding batchnorm at the end since it's something jh's done, i believe - ok, yeah, the curves immediately look better - maybe also look at diagnostics for the last layer - also can probably increase lr again - noticed that used a wrong model
##### tta
- write and export CapturePreds(Callback)
	- before_fit
	- after_batch
- fit the same learner for 1 val epoch with it
- do the same but now with augment (horizontal flip) during validation
- average the predictions and calculate accuracy
##### random erase
- write and show an example of a random erase (replace with noise that preserves statistics)
- what's wrong with images brightness? explain it
- write and export \_rand_erase1(x, pct, xm, xs, mn, mx) which does not have this problem
- show it
- write and export rand_erase(x, pct, max_num)
- show it
- write and export RandErase(nn.Module)
	- init(self, pct, max_num)
	- forward(self, x)
- fit a model with this additional transform for 50 epochs
##### random copy
- write and show an example of a random copy
- write and export \_rand_copy1(x, pct)
- write and export rand_copy(x, pct, max_num)
- show it
- write and export RandCopy(nn.Module)
	- init(self, pct, max_num), forward(self, x)
- fit 2 models with this transform for 25 epochs
- average their predictions and calculate accuracy
###### notes
- maybe also write a random shuffle-copy or random swap


### Homework

- create your own schedulers that work with PyTorch's optimizers
	- CosineAnnealing
	- OneCycle
	- make sure that they work correctly with the batch scheduler callback
- try beating on 5 (0.934, 0.219), 20 (0.943(0.945), 0.173), 50 (0.949, 0.163) epochs regimes TODO



---

## Lesson 19

- add diffusers to requirements


### 14_augment

##### Dropout
- show binomial
- write Dropout(nn.Module)
- write get_model with it (+nn.dropout2d)
- fit a model
###### notes
- what if instead of delete we do the same random erase/copy/shuffle TODO


### 15_ddpm

#### denoising diffusion probabilistic models with miniai
##### imports
- copy imports
- read ddpm paper
##### load the dataset
- load fashion mnist (lower bs, resize to 32 and don't normalize)
###### notes
- it makes sense to not normalize for ease of implementation but i think normalizing will improve things
##### create model
- import unet2dmodel from diffusers and create it (32, 64, 128, 128)
##### training - easy with a callback!
- write DDPMCB(TrainCB):
	- order after devicecb
	- init(self, n_steps, beta_min, beta_max) (setup variance schedule: beta, alpha, alpha_bar, sigma (formulas from the paper))
	- predict
	- before_batch (add noise:  epsilon, n, t, alpha_bar_t, xt -> ((xt, t), epsilon))
	- sample(self, model, sz) (reverse process: n steps do: t_batch, z, alpha_bar_t1, beta_bar_t, beta_bar_t1, noise_pred, x0_hat, x0_coeff, xt_coeff, x_t -> a sequence of images showing the process)
- fit the model
- save model
###### notes
- i am a bit confused about what was calculated in their code and what it says in the paper
- it's a bit slow for me so I will run (or at least iterate) with only 5000 training images and for 1 epoch
- ok, the sampling does not work - a bug or need to train for longer - ok while checking the code will train for 5 epochs
- well, both things didn't work out - let's just try on the full dataset since the loss is still a bit too high and that might explan the results
- trained with a bit large size - it's different? let's try just with one class - ok that worked - i guess really just needed more training
- still gonna look at each step and try to do the full sample algorithm and not the final one in the paper - yeah, that works much better
##### inference
- generate samples
- show final results
- show the sampling process


### 17_ddpm_v2

- copy 15_ddpm
- export as accel
- export imports
- split the callback into the parts (noisify, sample, from Callback) and write UNet(UNet2Model) to pack input and extract output
- show what happens to images through each step
- fit for one batch and show batch; also check out norm_num_groups in UNet2Model and use fewer channels
#### training
- write init_ddpm(model) (down block residuals as zero and downsamplers as orthogonal; up block residuals as zero; out conv as zero;or maybe copy it while still figuring it out)
- try fitting a smaller model (also be careful with opt's lower epsilon)
#### sampling
- show samples
### Homework
- implement dropout2d (drop all channels) TODO



---

## Lesson 20

- add accelerate, timm to requirements


### 17_ddpm_v2

#### mixed precision
- write collate_ddpm(b)
- write dl_ddpm(ds)
- get dls
- write and export MixedPrecision(TrainCB) (look at torch's mixed precision)
	- before_fit, before_batch, after_loss, backward, step
- fit the model (probably increase batch size -> also need largerer lr)
##### accelerate
- write and export AccelerateCB(TrainCB)
	- init, before_fit, backward
- rewrite noisify functionality to move device based on the input and return one tuple
- write DDPMCB2(Callback) (it just unpacks predicitons for others)
	- after_predict
- fit model (something with n_inp and don't use UNet wrapper)
###### notes
- oh, I already wrote it to move everything to correct device, nice
- didn't notice at first that should stop using the wrapper
- ok, I am having big problems with inputs... need to figure out what is going wrong with slices - ok, i actually needed to rewrite add noise but to just return one tuple instead of nesting that I was doing
- for some reason it's much slower - oh, noticed that I wasn't using mixed precision in the last training run and it's also slower - ok, gonna use a profiler - it seems to be something to do with running mixedprecision for a while (something with aten::\_local\_scalar\_dense) - it might to do with ProgressCB - no, even though the time taken is different, it's still just mixedprecision
- TODO actually figured out why mixedprecision is slower
#### a sneaky trick
- write MultDL (spits the same batch mult times)
	- init, len, iter


### 16a_styletransfer

#### setup
- copy imports
- get two image urls (face_url, spiderweb_url in nb)
#### loading imges
- write download_image(url) (can use fc.urlread and torchvision, also resize (256) and normalize)
- download content image and show it
- move the image to gpu
#### optimizing images
- write LengthDataset: init, len, getitem
- write get_dummy_dls(length)
- write TensorModel(nn.Module): init, forward
- show image using TensorModel and random noise in shape of the image
- write ImageOptCB(TrainCB): predict, get_loss
- write loss_fn_mse(im)
- fit the model to get content image from noise
- show the result
#### viewing progress
- write ImageLogCB(Callback)
- fit the model again with the CB
- show the optimization process
#### getting features from vgg16
##### load vgg network
- load vgg16 using timm
##### normalize images
- create imagenet_mean and std
- write normalize(im)
###### notes
- oh, i didn't need to normalize before that (or rather, will use imagenet normalization and to simplify we will do it on the fly)
##### get intermediate representations
- (hw) do it using hooks TODO
- write calc_features(imgs, target_layers=(18,25))
#### optimizing an image with content loss
- write ContentLossToTarget: init(self, target_im, target_layers), call(self, input_im)
- fit the model again
##### choosing the layers determines the kind of features that are important
- fit again but with earlier layers as targets
- (hw extra) optimize only for one of the filters TODO
#### style loss with gram matrix
##### trying it out
- load a style image
- write calc_grams(img, target_layers=(1,6,11,18,25))
- write StyleLossToTarget
- run it on the content image
#### style transfer
- fit the model and show the result
- (hw) TODO
	- play with target layers
	- starting from a random image
	- scaling between losses, number of steps, lr
	- tryout other networks
- (self) TODO combine losses to run the model only once
- (self) TODO check that everything is correct by running on the same images as in lessons
###### notes
- ok, it didn't work, lol - let's run each loss separately to see what would be the result - maybe it's just a bad style image - oh, i didn't start from a content image
- the style is not that visible so gonna increase multiplier for it in the total loss - oh, think it was just not completely clean style images


### 16b_nca

#### background: neural cellular automata
##### goal: match this style with an nca
- copy download_image and download a style image
##### style loss
- copy vgg16, normalize, calc_features, calc_grams (change it to be batched), StyleLossToTarget
- create style_loss with the style image
##### defining the nca model
- define small num_channels and hidden_n
- write make_grids(n, sz=128) (creates 'worlds')
- create filters (hard-coded, maybe can get them from the paper?)
- write perchannel_conv(x, filters) (with circular pad mode, applies the same weights for each layer)
- create grids (x) and do one step to get the input shape
- create a small linear model for that shape (2 layers, brain)
- flatten out the inputs (maybe using einops.rearrange)
- get brain predictions
- create a small conv model (so no need for rearrange)
- write SimpleCA(nn.Module):
	- init(self, zero_w2=True)
	- forward(self, x, update_rate=0.5) (do random update on only some of the cells)
	- to_rgb(self, x)
##### training
- copy LengthDataset, get_dummy_dls
- write NCAProgressCB(ProgressCB) (torchivision...make_grid, subplots, display, update)
	- after_batch(self, learn) (add grid preview and setup lims for loss)
- write NCACB(TrainCB)
	- init(self, ca, style_img_tensor, style_loss_scale=0.1, size=256, step_n_min=32, step_n_max=96) (sets up the pool)
	- predict(self, learn) (pick samples from pool and zero some out, apply the model some number of times, update the pool)
	- get_loss(self, learn) (style_loss and overflow_loss)
	- backward(self, learn) (with gradient normalization)
- setup model, cbs, loss and fit
- look at the final sample
- apply to a random grid and show the progress
- (hw) try with larger network
- (hw+) write a fragment shader (shadertoy.com) that runs nca
- TODO GET IT TO TRAIN PROPERLY (try making sure that variance is 1 and mean is 0)
###### notes
- ok, also needed to not use biases in the last layers
- why are the previews doubling at the end? - ah, ok i had the wrong name for the graph members in the progress bar
- the training is ok but does require lower learning rate to stop breaking - also changed the style scale loss since overflow was much lower comparatively - hm, still too low (a few orders of magnitude), i can multiply it by 1000 but might as well use sum instead of mean so that it's the 'same' for different sizes
why are there ca, style_img_tensor?
- it works slowly - trying to figure that out - ok, just had to reduce some numbers
- now, it's just very difficult to get it to train - i played around quite a bit with it and i think i get what's happening much more and it's actually very similar to how reinforcement learning is done - so maybe adding something like a replay buffer might help
- i think the problem might have been giving way too many channels (it makes sense to only give 3 channels for colors and 1 'communication' layer)

- TODO: check that correctly call model.train and model.eval


### Homework TODO

- (self) not sure if they do it later but writing own mixed precision might be fun
- play with other noise schedules
- try training / sampling with fewer steps
- play with other steps random selection
- (self) maybe increase the jump during training
- (self) what happens if we start from an existing image (in and out of distribution)



---

## Lesson 21

- add wandb, pytorch_fid to requirements


### 21_cifar10_and_wandb

- copy imports
- load cifar10 (all very like fashion mnist)
- write/copy linear_sched, args (alpha, beta), noisify like from ddpm_v2
- show prepared input
#### training
- unet, init_ddpm, dls, model (compare a smaller one to a default one with respect to the number of parameters)
- try training a default model for 1 epoch
- sample the current results
#### w&b cb
- write WandBCB(MetricsCB):
	- last in order
	- init(self, config, \*ms, project, \*\*metrics)
	- before_init (init wandb)
	- after_init (finish wandb run)
	- \_log(self, d) (log all metrics and sampled images)
	- sample_figure(self, learn)
	- after_batch (log loss)
- fit model for 10 epochs (or maybe still 5 or 3)
###### notes
- TODO should retrain with a lower batch size but a bigger model to see if i can get quality images (or maybe even add some kind of gradientaccumulation callback)


### 14_augment

#### augment 2
- train the final model but with normalization between -1 and 1


### 18_fid

- export as fid
#### fid
- copy imports
#### classifier
- load fashion_mnist (transform for diffusion)
- load the model from augmentations and create learner
- write append_outp(hook, mod, inp, outp) to store features
- get the features for a batch
- delete unnecessary layers
- .capture_preds to get
###### notes
- i think i should be padding instead of resizing since the size is so small and the border is mostly blank
- also should update HooksCallback to just accept modules
- i guess i've missed .capture_preds at some point
#### calc fid
- copy init args, noisify, collate_ddpm and dl_ddpm, UNet, sample
- create dataloaders
- load the ddpm model from ddpm_v2
- sample from the model
- capture preds for the samples images
- write and export \_sqrtm_newton_schulz(mat, num_iters=100) to calculate a matrix root (same as linalg.sqrtm)
- write and export \_calc_stats(feats) to get means and covariances
- write and export \_calc_fid(m1,c1,m2,c2)
- calculate fid for the samples images against real ones
- write and export \_squared_mmd(x, y)
- write and export \_calc_kid(x, y, maxs=50)
- calculate kid for the sampled images against real ones
###### notes
- maybe should increase the batch size used here
- TODO gonna skip writing \_sqrtm_newton_schulz
- should rename the sampled and predicted variables to be better
- i don't like the fid I am getting - ok, i think i am always getting the same value - ah, i used the wrong variable in \_calc_stats
- i don't understand what's maxs
#### fid class
- write and export ImageEval to perform fid/kid evaluation of samples:
	- init(self, model, dls, cbs=None)
	- get_feats(self, samp)
	- fid(self, samp)
	- kid(self, samp)
- create it
- get fid on samples
- get kid on samples
- plot how fid and kid change depending on the step of sampling
- get fid and kid on a batch of real images
###### notes
- for some reason now I am getting larger scores - I was adding 1 instead of subtracting
#### inception
- write IncepWrap(nn.Module) to convert 1channel images to work with Inception
- get fid and kid
###### notes
- oh, i only handle inputs with 2 dimensions


### 19_ddpm_v3

- copy imports
- load fashion_mnist but normalize to -0.5 0.5
- write linear_sched(betamin, betamax, n_steps) (returns an object with attributes for diffusion (alphsa, betas, sigmas))
- write abar(t, T) (squared cosine wave)
- write cos_sched(n_steps) (similar to linear_sched)
- plot alpha_bars to compare 
- plot slopes
- try to get linear_sched closer to cos_sched and then plot them and slopes
- repeat the same process from ddpm_v2 but train a larger model trained for longer (also time the sampling)
- run fid on them
###### notes
- wait, how do i get alphas
#### skip sampling
- write sample with some steps skipped
- run fid on them
- write sample2 trying to speed it up more
###### notes
- because of the schedule it's not complete clear how to skip - i guess just skipping the noise generation for each step should help

- TODO: maybe don't need to clip in sample (check the papers and fastai's nbs)


### 20_ddim

#### denoising diffusion implicit models - ddim
- copy exports
- load fashion_mnist
- import from diffusers DDIM and DDPM pipelines and schedulers
##### diffusers ddpm scheduler
- load model from ddpm_v3
- create scheduler
- create random batch and do 1 prediction step
- step the scheduler
- do a loop for all timestamps
- show samples
- calculate fid/kid (a batch of 2048 was stable enough)
###### notes
- i think there is something wrong with how the clipping is done - yeah, need to change clipping in scheduler - better but still something is wrong - also had to use fixed_large variance type - i had to check step by step and ended up looking at min values during the first step - i used uniform noise :) - maybe now - i can even revert to default setting and it will still work TODO check that
##### diffusers ddim scheduler
- create scheduler with 333 steps
- write diff_sample(model, sz, sched, \*\*kwargs) to samples
- show samples
- calculate fid/kid
- continue reducing and looking at fid/kid
###### notes
now ddim with 333 steps look rough - needed to not use the default noise weight (eta=0) and change it to 1
##### implementing ddim
- copy linear_sched
- write ddim_step(x_t, t, noise, abar_t, abar_t1, bbar_t, bbar_t1, eta)
- write sample(f, model, sz, n_steps, skips=1, eta=1.)
- show samples and fid/kid
###### notes
- i got really turn around here when reversing ts
- also, my implementation of ddim seems to be working much better - TODO figure out why



---

## Lesson 22

- add seaborn to requirements


### 22_cosine

#### cosine schedule
- copy exports
- load fashion_mnist
- write/copy abar(t), inv_abar(x), noisify(x0), collate_ddpm(b), dl_ddpm(ds), UNet, init_ddpm(model) 
- show how noisify works now
- fit the model with tihs now and save
- write denoise(x_t, noise, t)
- show what single step does using denoise
###### notes 
- i defintely got confused with the mins and maxs of images for different models and need to recheck them all TODO
#### sampling
- load model for feature extraction
- write/copy ddim_step(x_t, noise, abar_t, abar_t1, bbar_t, bbar_t1, eta, sig), sample(f, model, sz, steps, eta=1.)
- run evaluation on the trained model
###### notes
- it's difficult to make sure that the values don't blow up - i can't get this thing to work - might not have trained well enough but I don't have the patience to retrain/train more so will do that later TODO


### 22_noise-pred

#### predicting the noise level of noisy fashionmnist images
##### imports
- copy exports
##### load dataset
- load fashion_mnist
- write/copy noisify(x0), collate_ddpm(b), dl_ddpm(ds) (we are trying to predict t and not noise given t; also might transform t to be not only in \[0,1\]) 
- show a batch of data with targets
- get a baseline result when predicting average/'center' (can make a model that outputs a constant)
- write flat_mse(x,y)
- write/copy get_model(act, nfs, norm)
- fit model
- save model
- show predictions compared to targets
###### notes
just noticed that I was using double batch sized in dl_ddpm TODO fix those in previous nbs
i think i should increase the batch size
##### no-time model
- copy/write the ddpm training setup but without explicit t (with constnant t)
- fit model and save it
###### notes
- should actually make it possible to pass bs to dl_ddpm
##### sampling
- copy/write the ddim sampling setup but without explicit t
- calculate fid/kid
- it should be quite bad if still mostly the same, but we can use the t prediction model to limit assumed t vs real t
- recalculate fid/kid
- plot how fid/kid changes
###### notes
- ok got confused here, need to figure out what was actually done here but for now I will just do some things TODO figure out what actually needed to do


### 23_karras

following karras paper
#### karras pre-conditioning
- copy imports
- load fashion_mnist (-1 to 1)
- get std (0.33 works better?)
- write scalings(sig) to calculate c_skip, c_out  and c_in
- sample sigmas
- show histogram
- show kdeplot
- write/copy noisify(x0) but with scalings, collate_ddpm, dl_ddpm
- show batch to show the target interpolation
- show mean and var
- (TODO hw) try also making sure that mean is 0
##### train
- copy the rest of training setup and train
- save the model
- write denoise(target, noised_input)
- show what one denoising step is doing
- show 1 denoising step from pure noise
##### sampling
- write sigmas_karras(n, sigma_min=0.01, sigma_max=80, rho=7, device="cpu")
- plot sigmas
- write denoise(model, x, sig) (using scalings)
- write sample_euler(x, sigs, i, model) ("linear" addition)
- TODO write get_ancestral_step(sigma_from, sigma_to, eta=1.)
- TODO write sample_euler_ancestral(x, sigs, i, model, eta=1 (also adds random noise)
- TODO write sample_heun(x, sigs, i, model, s_churn=0., s_tmin=0., s_tmax=float("inf"), s_noise=1.) (average of 2 steps)
- write sample(sampler, model, steps=100, sigma_max=80, \*\*kwargs)
- sample, show and evaluate for each
- (TODO \*) write linear_multistep_coeff(order, t, i, j)
- (TODO \*) write sample_lms(model, steps=100, order=4, sigma_max) (uses previous slopes to estimate next step)
- (TODO \*) sample, show and evaluate lms
###### notes
- i think i've messed up one of those denoises - i think i've fixed it (forgot to multiply by c_in)



---

## Lesson 23


### 23_karras

##### sampling
- fix potential bug: that scaling is consistent for the evaluation model and data transforms TODO


### 24_imgnet_tiny

#### tiny imagenet
- copy imports
##### data processing
- copy url to tiny imagenet
- write TinyDS: init, len, getitem  (path and name)
- create training ds
- write TinyValDS
- create val ds
	- write TfmDS (adds transforms on top of a ds)
- create a map for id2str/str2id
- write tfmx(x) to read images with normalization (calculate later)
- write tfmy(y) to get ids
- create dss with transforms
- write denorm(x)
- calculate xmean and xstd
- create a datalaoder
- show an image from a batch
- read synsets and create an id2name
- show a batch with titles
- create dataloaders
##### basic model
- write tfm_batch
- create tfms
- copy/write get_dropmodel - the previous when with dropout
- look at batch
- train model (~59.3%, 25epochs) and save it
###### notes
- everything was going ok until I got an error in LRFinderCB during cleanup_fit because lrs and losses are of different size, huh? - i've used Learner instead of TrainLearner
##### deeper 
- write res_blocks(n_bk, ni, nf, stride, ks, act, norm) to allow several layers per downsampling block
- write get_dropmodel (~2-2.5x flops)
- train model (~61.8%, 25epochs) and save it
###### notes
- got an OOM during validation so recreated dls but with the same batch_size for all - also will probably reduce batch further for a larger model
##### more augmentation
- add more augmentation (trivialaugment might work poorly on a batch) 
- import PIL.Image
- write tfmx(x, aug) to load images now
- create dss and dls
- write conv, \_conv\_block, ResBlock, get_dropmodel, get_model to make a preactivation ResNet
- show a batch
- train model (~64.9%, 50 epochs) and save it
###### notes
- gonna figure out a way to do some augmentations using batchtransformcb
- just noticed that I was using train set for validation
- also, maybe augmentation are a bit too strong? - yeah, since they are the same for the whole batch it doesn't work that well so should try to do augmentations for each image separately TODO


### 24a_imgnet_tiny

- (\*) copy the previous one and train for 200 epochs (67.5%) TODO


### 25_superres

#### tiny imagenet
- copy imports
##### data processing
- copy/modify from previous but with target being the original image and the input a 2x downsampled and then rescaled back with nn image with same augments (except some erasing in input) and load data
- show batch
###### notes
- i guess i'll just move everything to dataset to simplify 
##### denoising autoencoder
- write up_block(ni, nf, ks, act, norm)
- write get_model(act, nfs=(32,64,...,1024), norm, drop)
- train model (~0.207, 5 epochs)
- show preds
##### unet
- write TinyUnet
- write zero_wgts(l)
- create and init model
- train model (~0.073, 20 epochs)
- show preds
###### notes
- ah, summary does not work when the whole model is not iterable - don't wanna fix it right now TODO
##### perceptual loss
- load the previous classifier
- leave only the first 3 resblocks
- write comb_loss(inp, tgt) (mse between images + mse between features)
- write get_unet
- train model (~0.303, 20 epochs) (why is it difficult to compare? can't we just calculate mse after the fact?) and save it
- show preds
###### notes
- for some reason i get oom during validation TODO (maybe valdiation workers?)
- also can just run mse separately to directly compare after training TODO
##### perceptual loss
- create model but initialize the down path with pretrained classifier and freeze them
- fit the model for 1 epoch
- unfreeze and train (~0.198, 20 epochs) and save
- show preds
###### notes
- i don't have an exact model that I can use but i'll improvise and see how that works
- ok, was getting a weird loss, but I just forgot to snip the classification model for the perceptual loss
- now's better, but now the loss blew up - will just lower lr for the first epoch
##### cross-convs
- write cross_conv (resblock + a conv)
- write TinyUnet but with cross_conv
- train with unfreezing (~0.189, 1+20 epochs) and save
###### notes
- again got a spike in loss but now in main training stage


### homework

- TODO style transfer with unets
- TODO colorization with unets
- TODO remove jpeg artifacts
- TODO improve superres



---

## Lesson 24


### 26_diffusion_unet

#### diffusion unet
- copy imports
- load fashion_mnist
- copy ddpm training setup
##### train
Based on diffusers (preactivation)
- write unet_conv(ni, nf, ks=3, stride=1, act=nn.SiLU, norm=None, bias=True)
- write UnetResBlock(nn.Module):
	- init(self, ni, nf=None, ks=3, act=nn.SiLU, norm=nn.BatchNorm2d)
	- forward
- write SaveModule():
	- forward(self, x, args, kwargs)
- write SavedResBlock(SaveModule, UnetResBlock)
- write SavedConv(SaveModule, nn.Conv2d)
- write down_block(ni, nf, add_down=True, num_layers=1)
- write upsample(nf)
- write UpBlock(nn.Module):
	- init(self, ni, prev_nf, nf, add_up=True, num_layers=2)
	- forward(self, x, ups)
- write UNet2DModel(nn.Module): (based on StableDiffusion)
	- init(self, in_channels=3, out_channels=3, nfs=(224,448,672,896), num_layers=1)
	- forward
- create model (in=out=1, nfs=(32,64,128,256), n=2)
- fit the model (25 epochs)
###### notes
- maybe you can approximate SiLU and make it faster TODO
- i just can't figure out those shapes for some reason to correctly count the required number of filters - ok, after a long time trying to figoure it out, I think I just messed up the number of downblocks
- that's a very weird fall i got in the training loss? why does it look like i had a sudden decrease in lr
##### timestamps
- create emb_dim, tsteps, max_period (16, (-10,10), 10000, time_embedding args, sinusoidal)
- show images of t_emb
- write timestamp_embedding(tsteps, emb_dim, max_period)
##### timestep model
- write lin(ni, nf, act=nn.SiLU, norm=None, bias=True)
- write EmbResBlock(nn.Module): (once again based on StableDiffusion (embedding is applied as scale and shift))
	- init(self, n_emb, ni, nf=None, ks=3, act=nn.SiLU, norm=nn.BatchNorm2d)
- write saved(m, blk) (wraps m.forward to save)
- write DownBlock:
	- init, forward
- write UpBlock:
	- init, forward
- write EmbUNetModel:
	- init, forward
- create model and fit it (~0.137, 25epochs)
###### notes
- for some reason I am getting the final output saved 2 times - because for some reason I wrapped the downblock, too
that's a really big jump in validation loss lol
- oh, because of that wraps(?) you can't pickle the model, hm - oh, well
##### sampling
- copy sampling and eval setup and calculate fid/kid
###### notes
- for some reason sampling for more steps makes it not work TODO


### 27_attention

- write SelfAttention: (based on StableDiffusion (a bit ineffective/hacky?), 1d attention with addition) (import AttentionBlock from diffusers to compare to)
	- init(self, ni)
	- forward
- write cp_params(a,b) to copy weights for comparision
- now show how to use only 1 linear instead of 3
- write heads_to_batch(x, heads)
- write batch_to_heads(x, heads)
- write SelfAttentionMultihead:
	- init(self, ni, nheads)
	- forward
###### notes
- oh, they changed AttentionBlock to Attention and it works differently now


### 28_diffusion-attn-cond

- export as diffusion
#### diffusion net
- copy imports
- write/copy and export:
	- abar(t)
	- inv_abar(t)
	- noisify(x0)
	- collate_ddpm(b)
	- dl_ddpm(ds)
- load fashion_mnist
##### train
- write/copy and export:
	- timestamp_embedding(tsteps, emb_dim, max_period)
	- pre_conv(ni, nf, ks, stride, act, norm, bias)
	- upsample(nf)
	- SelfAttention
	- SelfAttention2D(SelfAttention)
	- EmbResBlock
	- saved
	- DownBlock
	- UpBlock
	- EmbUNetModel (attn_channels=8, attn_start=1)
- create model and fit it (~0.033, 25epochs)
###### notes
- gonna use SaveModule instead of saved
- not fitting into memory at all - dramatically decreased the batch size (TODO add grad accum or is there something wrong with SelfAttention) - 5GB with heads=16 and 1.5 GB with nheads=0 - oh, i see: the memory depends on the size of each head so making it constant is the usual way to do, so that's why - attn_channels instead of nheads
- again a sudden spike but now in training loss - change groupnorm to layer norm and added/moved norm and act to where they were missing and reduced lr, also increased the max period in timestamp_embedding -TODO (figure out how to increase lr) - those didn't really work so i also checked the init and maybe it's wrong here so turned it off and it seems to be much better
- TODO retrain previous without init and also come up with better init
- TODO in 22_cosine I am loading the wrong model (19_ddpm_v3_lincos_10 instead of 22_cosine_10)
- TODO can backtrack fixes from here to cosine (the error was me forgetting to switch dtype from int64 to float for ts)
##### sampling
- copy sampling setup and evaluate and export:
	- ddim_step
	- sample
###### notes
- worse than without attention TODO
##### conditional model
- write collate_ddpm(b) (also adds label)
- write CondUNetModel:
	- init(self, n_classes, in_channels, out_channels, num_layers)
	- forward
- create model and fit it (~0.033, 25epochs)
- write and export cond_sample(c, f, model, sz, steps, eta=1.)
- show samples



---

## Lesson 25

### simple_diffusion_audio

- (\*) recreate this on tglcourse/5s_birdcallsamples_top20 (~0.034, 15 epochs) TODO
#### loading audio dataset
#### dataloaders
#### model + training
#### sampling


### 29_vae

#### vae
- copy imports
- load fashion_mnist (but flatten)
##### autoencoder
- write lin(ni, nf, act, norm, bias)
- write init_weights
- write Autoenc: (784, 400, 200)
	- init, forward
- create and train model (~0.260, 20 epochs, bce loss) 
###### notes
- isn't using bce not good because of the min and max values TODO just do leaks in two directions
###### autoencoder sample
- compare output
- decode noise
##### vae
- write VAE:
	- init, forward
- write kld_loss
- (self, TODO) maybe check that variance will become 0
- write bce_loss
- (self, TODO) why not use just lv squared too?
- write vae_loss
- write FuncMetric(Mean):
	- init, update
- create and train model (~0.310 + ~0.031,20 epochs)
###### notes
- for some reason my FuncMetric is not calculating correctly - oh, i was assigning instead of incrementing
###### vae sample
- compare output
- decode noise
###### notes
- it's not working with noise - just like before I used rand instead of randn


### 30_lsun_diffusion-latents

#### lsun bedrooms
- copy imports
##### data processing
- load dataset (get url from nb)
- show a batch of images (256x256)
##### vae
- load pretrained "stabilityai/sd-vae-ft-ema"
- get latents from the batch and shoe them as images (use mean)
- show decoded images
- save latents to a memory mapped numpy file
###### notes
i think there is something about not using only mean
##### noisify
- rewrite/copy the process from previous and show intermediate steps
###### notes
- wait, did i forget to shuffle train for previous models?? TODO
##### train
- rewrite/copy the process from previous and show intermediate steps (~0.243, 25 epochs (took several hours), also add zero initialization of last convs in each block)
###### notes
- i am getting really large validation loss
- oh, i've messed up creating latents and was always overwriting the first batch
- also should probably add init - yeah, that helped a bit but there was also an increase in validation error - i wonder why those happen
and also should fix the names there: blocks vs layers in up vs down
##### sampling
- rewrite/copy the process from previous and show intermediate steps


### 31_imgnet_latents-widish

#### full imagenet
- (hw) use their notebook as a bootstrap and try to get goo classification results using latents
