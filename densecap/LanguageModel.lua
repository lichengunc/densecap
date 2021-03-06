require 'nn'
require 'torch-rnn'

local utils = require 'densecap.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(opt)
  parent.__init(self)

  opt = opt or {}
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.image_vector_dim = utils.getopt(opt, 'image_vector_dim')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.idx_to_token = utils.getopt(opt, 'idx_to_token')
  self.dropout = utils.getopt(opt, 'dropout', 0)

  local W, D = self.input_encoding_size, self.image_vector_dim
  local V, H = self.vocab_size, self.rnn_size

  -- For mapping from image vectors to word vectors
  self.image_encoder = nn.Sequential()
  self.image_encoder:add(nn.Linear(D, W))
  self.image_encoder:add(nn.ReLU(true))
  self.image_encoder:add(nn.View(1, -1):setNumInputDims(1))
  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2

  -- For mapping word indices to word vectors
  local V, W = self.vocab_size, self.input_encoding_size
  self.lookup_table = nn.LookupTable(V + 2, W)
  
  -- Change this to sample from the distribution instead
  self.sample_argmax = true

  -- self.rnn maps wordvecs of shape N x T x W to word probabilities
  -- of shape N x T x (V + 1)
  self.rnn = nn.Sequential()
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    --[[
    For Justin's torch-rnn, the LSTM takes input
    - c0: Initial cell state, (N, H)
    - h0: Initial hidden state, (N, H)
    - x : Input Sequence, (N, T, W)
    and output
    - h : Sequence of hidden states, (N, T, H)
    ]]
    self.rnn:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn:add(nn.Dropout(self.dropout))
    end
  end

  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)
  self.rnn:add(self.view_in)
  self.rnn:add(nn.Linear(H, V + 1))
  self.rnn:add(self.view_out)

  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  local parallel = nn.ParallelTable()
  parallel:add(self.image_encoder)
  parallel:add(self.start_token_generator)  -- bug here, non-sense at all
  parallel:add(self.lookup_table)
  self.net:add(parallel)
  self.net:add(nn.JoinTable(1, 2))
  self.net:add(self.rnn)

  self:training()
end


--[[
Decodes a sequence into a table of strings

Inputs:
- seq: tensor of shape N x T

Returns:
- captions: Array of N strings
--]]
function LM:decodeSequence(seq)
  local delimiter = ' '
  local captions = {}
  local N, T = seq:size(1), seq:size(2)
  for i = 1, N do
    local caption = ''
    for t = 1, T do
      local idx = seq[{i, t}]
      if idx == self.END_TOKEN or idx == 0 then break end
      if t > 1 then
        caption = caption .. delimiter
      end
      caption = caption .. self.idx_to_token[idx]
    end
    table.insert(captions, caption)
  end
  return captions
end


function LM:updateOutput(input)
  self.recompute_backward = true
  local image_vectors = input[1]
  local gt_sequence = input[2]
    
  if gt_sequence:nElement() > 0 then
    -- Add a start token to the start of the gt_sequence, and replace
    -- 0 with NULL_TOKEN
    local N, T = gt_sequence:size(1), gt_sequence:size(2)
    self._gt_with_start = gt_sequence.new(N, T + 1)
    self._gt_with_start[{{}, 1}]:fill(self.START_TOKEN)
    self._gt_with_start[{{}, {2, T + 1}}]:copy(gt_sequence)
    local mask = torch.eq(self._gt_with_start, 0)
    self._gt_with_start[mask] = self.NULL_TOKEN
    
    -- Reset the views around the nn.Linear
    self.view_in:resetSize(N * (T + 2), -1)
    self.view_out:resetSize(N, T + 2, -1)
    self.output = self.net:updateOutput{image_vectors, self._gt_with_start}
    self._forward_sampled = false
    return self.output
  else
    self._forward_sampled = true
    if self.beam_size ~= nil then
      print('running beam search with beam size of ' .. self.beam_size)
      self.output = self:beamsearch(image_vectors, self.beam_size)
      return self.output
    else
      return self:sample(image_vectors)
    end
  end
end


--[[
Convert a ground-truth sequence of shape to a target suitable for the
TemporalCrossEntropyCriterion from torch-rnn.

Input:
- gt_sequence: Tensor of shape (N, T) where each element is in the range [0, V];
  an entry of 0 is a null token.
--]]
function LM:getTarget(gt_sequence)
  -- Make sure it's on CPU since we will loop over it
  local gt_sequence_long = gt_sequence:long()
  local N, T = gt_sequence:size(1), gt_sequence:size(2)
  local target = torch.LongTensor(N, T + 2):zero()
  target[{{}, {2, T + 1}}]:copy(gt_sequence)
  for i = 1, N do
    for t = 2, T + 2 do
      if target[{i, t}] == 0 then
        -- Replace the first null with an end token
        target[{i, t}] = self.END_TOKEN
        break
      end
    end
  end
  return target:type(gt_sequence:type())
end
--[[
image_vectors: N x D
--]]
function LM:beamsearch(image_vectors, beam_size)
  beam_size = beam_size or 20
  local Done_beams = {}
  local N, T = image_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  if image_vectors:type() == 'torch.CudaTensor' then
    seq = seq:cuda()
  end
  local lsm = nn.LogSoftMax():type(image_vectors:type())

  -- Find all the LSTM layers in the RNN
  local lstm_layers = {}
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      table.insert(lstm_layers, layer)
    end
  end

  -- Run each image vector separately, using the minibatch
  -- dimension to hold beams
  for i = 1, N do

    local done_beams = {}

    -- Reset states in the RNN
    for _, layer in ipairs(lstm_layers) do
      layer:resetStates()
      layer.remember_states = true
    end

    -- For the first two timesteps (image and START token) we will use
    -- a minibatch size of 1, so tell the views to expect 1 element
    self.view_in:resetSize(1, -1)
    self.view_out:resetSize(1, 1, -1)

    -- Encode the image vector and feed it to the RNN
    local image_vec = image_vectors[{{i, i}}]
    local image_vec_encoded = self.image_encoder:forward(image_vec)
    self.rnn:forward(image_vec_encoded)

    -- Feed a START token to the RNN
    local start = torch.LongTensor(1, 1):fill(self.START_TOKEN)
    local start_vec = self.lookup_table:forward(start)
    local scores = self.rnn:forward(start_vec):view(1, -1)

    -- Initialize our beams to the words with the highest logprobs
    local beams = seq.new(beam_size, T):fill(1)  -- (beam_size, T)
    local all_logprobs = lsm:forward(scores)
    local beam_logprobs, idx = torch.topk(all_logprobs, beam_size, 2, true)  -- (1, beam_size)
    beams[{{}, 1}] = idx

    -- Go into each LSTM layer and duplicate the cell and hidden states for
    -- all beams
    for _, layer in ipairs(lstm_layers) do
      local H = layer.cell:size(3)
      layer.cell = layer.cell:expand(beam_size, 1, H):clone()
      layer.c0:resize(beam_size, H):zero()
      layer.output = layer.output:expand(beam_size, 1, H):clone()
      layer.h0:resize(beam_size, H):zero()
    end

    -- For subsequent timesteps we will run with a batch size of beam_size,
    -- so reset the views so they expect it
    self.view_in:resetSize(beam_size, -1)
    self.view_out:resetSize(beam_size, 1, -1)

    for t = 2, T do
      local words = beams[{{}, {t - 1, t - 1}}] 
      local wordvecs = self.lookup_table:forward(words)
      local scores = self.rnn:forward(wordvecs):view(beam_size, -1)  -- (beam_size, V+1)
      local next_word_logprobs = lsm:forward(scores)

      -- If a beam already has an END token then any subsequent words should
      -- not contribute to its logprobs, so set them to zero
      local end_mask = torch.eq(torch.eq(beams, self.END_TOKEN):sum(2), 0)
      end_mask = end_mask:type(next_word_logprobs:type())
      next_word_logprobs:cmul(end_mask:expandAs(next_word_logprobs))

      -- For each beam, find the top beam_size next words
      local top_next_word_logprobs, word_idx
        = torch.topk(next_word_logprobs, beam_size, 2, true) -- (beam_size, beam_size)

      local beam_logprobs_dup = beam_logprobs:view(-1, 1)
                                  :expand(beam_size, beam_size)
                                  :contiguous()
                                  :view(beam_size * beam_size)
      local all_next_logprobs = top_next_word_logprobs:view(-1)
                                   + beam_logprobs_dup
      beam_logprobs, idx = torch.topk(all_next_logprobs, beam_size, 1, true)

      local all_next_beams = beams:view(beam_size, 1, T)
                               :expand(beam_size, beam_size, T)
                               :contiguous()
                               :view(beam_size * beam_size, T)
      all_next_beams[{{}, t}] = word_idx:view(-1)
      beams = all_next_beams:index(1, idx)

      for _, layer in ipairs(lstm_layers) do
        local H = layer.cell:size(3)
        local cell_dup = layer.cell:expand(beam_size, beam_size, H)
                                   :contiguous()
                                   :view(beam_size * beam_size, 1, H)
        layer.cell = cell_dup:index(1, idx)
        local out_dup = layer.output:expand(beam_size, beam_size, H)
                                    :contiguous()
                                    :view(beam_size * beam_size, 1, H)
        layer.output = out_dup:index(1, idx)
      end

      -- add to done_beams
      local end_mask = torch.gt(torch.eq(beams, self.END_TOKEN):sum(2), 0)
      for eix = 1, end_mask:nElement() do
        if end_mask[eix][1] == 1 then
          -- <end> found for this beam
          table.insert(done_beams, {seq = beams[eix], 
                                    logp = beam_logprobs[eix], 
                                    logppl = -beam_logprobs[eix]/(t-1)})
          -- kill this beam by making its logprobs super small
          beam_logprobs[eix] = -1000
        end
      end
      -- if all beams get to <end>, we jump out
      if end_mask:sum() == self.beam_size then goto done end

    end

    ::done::
    -- sort beams according to logppl or logp
    local function compare_logp(a,b) return a.logp > b.logp end -- used downstream
    local function compare_logppl(a, b) return a.logppl < b.logppl end  -- used upstream
    table.sort(done_beams, compare_logppl)
    seq[i] = done_beams[1].seq

    -- add to Done_beams
    local topk = math.min(self.beam_size, #done_beams)
    Done_beams[i] = {seq = torch.Tensor(topk, T):typeAs(beams),
                     logp = torch.Tensor(topk),
                     logppl = torch.Tensor(topk):typeAs(beam_logprobs)}
    for k = 1, self.beam_size do
      Done_beams[i].seq[k] = done_beams[k].seq
      Done_beams[i].logp[k] = done_beams[k].logp
      Done_beams[i].logppl[k] = done_beams[k].logppl
    end

  end

  for _, layer in ipairs(lstm_layers) do
    layer:resetStates()
    layer.remember_states = false
  end

  return seq, Done_beams
end


function LM:sample(image_vectors)
  local N, T = image_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local softmax = nn.SoftMax():type(image_vectors:type())
  
  -- During sampling we want our LSTM modules to remember states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)

  -- First timestep: image vectors, ignore output
  local image_vecs_encoded = self.image_encoder:forward(image_vectors)
  self.rnn:forward(image_vecs_encoded)

  -- Now feed words through RNN
  for t = 1, T do
    local words = nil
    if t == 1 then
      -- On the first timestep, feed START tokens
      words = torch.LongTensor(N, 1):fill(self.START_TOKEN)
    else
      -- On subsequent timesteps, feed previously sampled words
      words = seq[{{}, {t-1, t-1}}]
    end
    local wordvecs = self.lookup_table:forward(words)
    local scores = self.rnn:forward(wordvecs):view(N, -1)
    local idx = nil
    if self.sample_argmax then
      _, idx = torch.max(scores, 2)
    else
      local probs = softmax:forward(scores)
      idx = torch.multinomial(probs, 1):view(-1):long()
    end
    seq[{{}, t}]:copy(idx)
  end

  -- After sampling stop remembering states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end

  self.output = seq
  return self.output
end


function LM:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput)
  end
  return self.gradInput
end


function LM:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end


function LM:backward(input, gradOutput, scale)
  assert(self._forward_sampled == false, 'cannot backprop through sampling')
  assert(scale == nil or scale == 1.0)
  self.recompute_backward = false
  local net_input = {input[1], self._gt_with_start}
  self.gradInput = self.net:backward(net_input, gradOutput, scale)
  self.gradInput[2] = input[2].new(#input[2]):zero()
  return self.gradInput
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  parent.training(self)
  self.net:training()
end


function LM:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


function LM:clearState()
  self.net:clearState()
end


function LM:parameters()
  return self.net:parameters()
end
------------------------------------------------------------------------
-- Licheng's layers
------------------------------------------------------------------------
function LM:sample_with_hidden(image_vectors)
  local N, T = image_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local endH = torch.zeros(N, self.rnn_size):type(image_vectors:type())
  local softmax = nn.SoftMax():type(image_vectors:type())
  
  -- During sampling we want our LSTM modules to remember states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)

  -- First timestep: image vectors, ignore output
  local image_vecs_encoded = self.image_encoder:forward(image_vectors)
  self.rnn:forward(image_vecs_encoded)

  -- Now feed words through RNN
  for t = 1, T do
    local words = nil
    if t == 1 then
      -- On the first timestep, feed START tokens
      words = torch.LongTensor(N, 1):fill(self.START_TOKEN)
    else
      -- On subsequent timesteps, feed previously sampled words
      words = seq[{{}, {t-1, t-1}}]
    end
    local wordvecs = self.lookup_table:forward(words)
    local scores = self.rnn:forward(wordvecs):view(N, -1)
    local idx = nil
    if self.sample_argmax then
      _, idx = torch.max(scores, 2)
    else
      local probs = softmax:forward(scores)
      idx = torch.multinomial(probs, 1):view(-1):long()
    end
    seq[{{}, t}]:copy(idx)
    -- check if we just produced <end> for the first time
    for n = 1, N do
      if seq[{ n, t }] == self.END_TOKEN and torch.sum(endH[n]) == 0 then
        endH[n] = self.rnn:get(self.num_layers*2-1).output[{ n, 1, {} }]
      end
    end
  end

  -- After sampling stop remembering states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end

  return {seq, endH}
end
--[[
input:
- image_vectors: (N, 4096)
- seq: (N, T)
output:
- endH: (N, rnn_size)
]]
function LM:extract_hidden(image_vectors, seq)
  local N = image_vectors:size(1)
  assert(N == seq:size(1), 'image vectors and seq are not of same batch size.')
  local T = self.seq_length
  local endH = torch.zeros(N, self.rnn_size):type(image_vectors:type())

  -- During forward propagation, we want LSTM to remember state
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)

  -- First timestep: image vectors, ignore output
  local image_vecs_encoded = self.image_encoder:forward(image_vectors)
  self.rnn:forward(image_vecs_encoded)

  -- Now feed words through RNN
  for t = 1, T do
    local words = nil
    if t == 1 then
      -- On the first timestep, feed START token
      words = torch.LongTensor(N, 1):fill(self.START_TOKEN)
    else
      words = seq[{ {}, {t-1, t-1} }]
    end
    local wordvecs = self.lookup_table:forward(words)
    self.rnn:forward(wordvecs)
    -- check if we reach <end> for the first time
    for n = 1, N do
      if seq[{n,t}] == self.END_TOKEN and torch.sum(endH[n]) == 0 then
        endH[n] = self.rnn:get(self.num_layers*2-1).output[{ n, 1, {}}]
      end
    end
  end
  -- After forwarding, stop remembering states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end
  -- return
  return endH
end









