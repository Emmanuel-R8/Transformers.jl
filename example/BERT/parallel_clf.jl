using Distributed
addprocs(4)

@everywhere using Transformers
@everywhere using Transformers.Basic
@everywhere using Transformers.Pretrain
@everywhere using Transformers.Datasets
@everywhere using Transformers.Datasets: GLUE
@everywhere using Transformers.BidirectionalEncoder

@everywhere using Flux
@everywhere using Flux: onehotbatch, gradient
import Flux.Optimise: update!
@everywhere using WordTokenizers

include("./args.jl")

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

const Epoch = 20
const Batch = 4

if args["task"] == "cola"
  const task = GLUE.CoLA()

  @everywhere markline(sent) = ["[CLS]"; sent; "[SEP]"]
  @everywhere function preprocess(batch, labels, vocab, wordpiece, tokenizer)
    sentence = markline.(wordpiece.(tokenizer.(batch[1])))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)

    label = onehotbatch(batch[2], labels)
    return (tok=tok, segment=segment), label, mask
  end
else
  if args["task"] == "mnli"
    const task = GLUE.MNLI(false)
  elseif args["task"] == "mrpc"
    const task = GLUE.MRPC()
  end
  @everywhere markline(s1, s2) = ["[CLS]"; s1; "[SEP]"; s2; "[SEP]"]
  @everywhere function preprocess(batch, labels, vocab, wordpiece, tokenizer)
    s1 = wordpiece.(tokenizer.(batch[1]))
    s2 = wordpiece.(tokenizer.(batch[2]))
    sentence = markline.(s1, s2)
    mask = getmask(sentence)
    tok = vocab(sentence)

    segment = fill!(similar(tok), 1)
    for (i, sent) ∈ enumerate(sentence)
      j = findfirst(isequal("[SEP]"), sent)
      if j !== nothing
        @view(segment[j+1:end, i]) .= 2
      end
    end

    label = onehotbatch(batch[3], labels)
    return (tok=tok, segment=segment), label, mask
  end
end

const labels = get_labels(task)

const _bert_model, wordpiece, tokenizer = pretrain"Bert-uncased_L-12_H-768_A-12"
const vocab = Vocabulary(wordpiece)

const hidden_size = size(_bert_model.classifier.pooler.W ,1)
const clf = gpu(Chain(
  Dropout(0.1),
  Dense(hidden_size, length(labels)),
  logsoftmax
))

const bert_model = gpu(
  set_classifier(_bert_model,
                 (
                   pooler = _bert_model.classifier.pooler,
                   clf = clf
                 )
                 )
)

const ps = params(bert_model)
const opt = ADAM(1e-4)

function acc(p, label)
  pred = Flux.onecold(collect(p))
  label = Flux.onecold(collect(label))
  sum(pred .== label) / length(label)
end


function loss(data, label, mask=nothing)
  e = bert_model.embed(data)
  t = bert_model.transformers(e, mask)

  p = bert_model.classifier.clf(
    bert_model.classifier.pooler(
      t[:,1,:]
    )
  )

  l = Basic.logcrossentropy(label, p)
  return l, p
end

function train!()
  global Batch
  global Epoch
  @info "start training: $(args["task"])"
  for e = 1:Epoch
    @info "epoch: $e"

    remotecall_fetch(2) do
      dataset
    end
    datas = RemoteChannel(()->Datasets.merges!(dataset(Train, task)), 2)
    pipes = Transformers.Datasets.separate!(datas, [3,4,5]; csize=20)
    Datasets.stage!(Datasets.Take, Datasets.batches!, pipes, Batch)
    Datasets.stage!(Transformers.Datasets.batched, pipes)
    Datasets.stage!(preprocess, pipes, labels, vocab, wordpiece, tokenizer)
    datas = Datasets.collect!(pipes, 1; csize=1000)
    datas = channel_from_id(remoteref_id(datas))
    datas = Datasets.funnel(Datasets.Put, todevice, datas; csize=1000)

    i = 1
    al::Float64 = 0.
    # while (batch = get_batch(datas, Batch)) !== nothing
    for batch in datas
      data, label, mask = batch
      l, p = loss(data, label, mask)
      @show l
      a = acc(p, label)
      al += a
      grad = gradient(()->l, ps)
      i+=1
      update!(opt, ps, grad)
      i%16==0 && @show al/i
    end

    test()
  end
end

function test()
  Flux.testmode!(bert_model)
  i = 1
  al::Float64 = 0.
  datas = dataset(Dev, task)
  while (batch = get_batch(datas, Batch)) !== nothing
    data, label, mask = todevice(preprocess(batch))
    _, p = loss(data, label, mask)
    # @show l
    a = acc(p, label)
    al += a
    i+=1
  end
  al /= i
  Flux.testmode!(bert_model, false)
  @show al
end

train!()
