import os
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd

import torch
import einops
import datasets
from torch.utils.data import DataLoader
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from utils import adjust_precision


def process_activation_batch(args, batch_activations, step, batch_mask=None):
    cur_batch_size = batch_activations.shape[0]

    if args.activation_aggregation is None:
        # only save the activations for the required indices
        batch_activations = einops.rearrange(
            batch_activations, 'b c d -> (b c) d')  # batch, context, dim
        processed_activations = batch_activations[batch_mask]

    if args.activation_aggregation == 'last':
        last_ix = batch_activations.shape[1] - 1
        batch_mask = batch_mask.to(int)
        last_entity_token = last_ix - \
            torch.argmax(batch_mask.flip(dims=[1]), dim=1)
        d_act = batch_activations.shape[2]
        expanded_mask = last_entity_token.unsqueeze(-1).expand(-1, d_act)
        processed_activations = batch_activations[
            torch.arange(cur_batch_size).unsqueeze(-1),
            expanded_mask,
            torch.arange(d_act)
        ]
        assert processed_activations.shape == (cur_batch_size, d_act)
    
    elif args.activation_aggregation == 'mean':
        # average over the context dimension for valid tokens only
        masked_activations = batch_activations * batch_mask
        batch_valid_ixs = batch_mask.sum(dim=1)
        processed_activations = masked_activations.sum(
            dim=1) / batch_valid_ixs[:, None]

    elif args.activation_aggregation == 'max':
        # max over the context dimension for valid tokens only (set invalid tokens to -1)
        batch_mask = batch_mask[:, :, None].to(int)
        # set masked tokens to -1
        masked_activations = batch_activations * batch_mask + (batch_mask - 1)
        processed_activations = masked_activations.max(dim=1)[0]

    return processed_activations

def make_token_dataset(text, image_paths, processor, tokenizer):
    ids, masks, pixels = [], [], []
    
    for image_path in tqdm(image_paths, desc='Tokenizing images'): 
        prompt = [
            Image.open(image_path),
            # "Question: What's on the picture? Answer:",
            text,
        ]

        inputs = processor(prompt, return_tensors="pt")
        token_ids = inputs.input_ids

        # add bos token
        token_ids = torch.cat([
            torch.ones(token_ids.shape[0], 1,
                    dtype=torch.long) * tokenizer.bos_token_id,
            token_ids], dim=1
        )

        prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
        entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
        entity_mask[:, prompt_tokens] = False
        entity_mask[token_ids == tokenizer.pad_token_id] = False

        ids.append(token_ids.tolist())
        masks.append(entity_mask.tolist())
        pixels.append(inputs.pixel_values)
    
    dataset = datasets.Dataset.from_dict({
        'input_ids': ids,
        'entity_mask': masks,
        'pixel_values': pixels
    })

    dataset.set_format(type='torch', columns=['input_ids'])
    return dataset


@torch.no_grad()
def get_layer_activations_hf(
    args, model, tokenized_dataset, layers='all', device=None,
):
    if layers == 'all':
        layers = list(range(model.config.num_hidden_layers))
    if device is None:
        device = model.device

    entity_mask = torch.tensor(tokenized_dataset['entity_mask'])

    n_seq, ctx_len = tokenized_dataset['input_ids'].shape
    activation_rows = entity_mask.sum().item() \
        if args.activation_aggregation is None \
        else n_seq

    layer_activations = {
        l: torch.zeros(activation_rows, model.config.hidden_size,
                       dtype=torch.float16)
        for l in layers
    }
    assert args.activation_aggregation == 'last'  # code assumes this
    offset = 0
    bs = args.batch_size
    dataloader = DataLoader(
        list(zip(tokenized_dataset['input_ids'], tokenized_dataset.get('pixel_values', []))), 
        batch_size=bs, 
        shuffle=False
    )

    for step, (input_batch, pixel_batch) in enumerate(tqdm(dataloader, disable=False)):
        # clip batch to remove excess padding
        batch_entity_mask = entity_mask[step*bs:(step+1)*bs]
        last_valid_ix = torch.argmax(
            (batch_entity_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
        input_batch = input_batch[:, :last_valid_ix].to(device)
        batch_entity_mask = batch_entity_mask[:, :last_valid_ix]

        if len(pixel_batch) > 0:
            pixel_batch = pixel_batch.to(device)
            out = model(
                input_ids=input_batch, 
                pixel_values=pixel_batch, 
                output_hidden_states=True,
                output_attentions=False, 
                return_dict=True, 
                use_cache=False
            )
        else:
            out = model(
                input_ids=input_batch,
                output_hidden_states=True,
                output_attentions=False, 
                return_dict=True, 
                use_cache=False
            )

        # do not save post embedding layer activations
        for lix, activation in enumerate(out.hidden_states[1:]):
            if lix not in layer_activations:
                continue
            activation = activation.cpu().to(torch.float16)
            processed_activations = process_activation_batch(
                args, activation, step, batch_entity_mask
            )

            save_rows = processed_activations.shape[0]
            layer_activations[lix][offset:offset + save_rows] = processed_activations

        offset += input_batch.shape[0]

    return layer_activations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--model', default='HuggingFaceM4/idefics-80b',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--entity_type',
        help='Name of entity_type (should be dir under data/entity_datasets/)')
    parser.add_argument(
        '--activation_aggregation', default='last',
        help='Average activations across all tokens in a sequence')
    # base experiment params
    parser.add_argument(
        '--device', default="cuda" if torch.cuda.is_available() else "cpu",
        help='device to use for computation')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size to use for model.forward')
    parser.add_argument(
        '--save_precision', type=int, default=8, choices=[8, 16, 32],
        help='Number of bits to use for saving activations')
    parser.add_argument(
        '--n_threads', type=int,
        default=int(os.getenv('SLURM_CPUS_PER_TASK', 8)),
        help='number of threads to use for pytorch cpu parallelization')
    parser.add_argument(
        '--layers', nargs='+', type=int, default=None)
    parser.add_argument(
        '--use_tl', action='store_true',
        help='Use TransformerLens model instead of HuggingFace model')
    parser.add_argument(
        '--is_test', action='store_true')
    parser.add_argument(
        '--prompt_name', default='all')

    args = parser.parse_args()

    print('Begin loading model')
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = IdeficsForVisionText2Text.from_pretrained(args.model, 
                                                      quantization_config=quantization_config, 
                                                      device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print('Finished loading model')

    torch.set_grad_enabled(False)
    model.eval() # check if this is necessary

    df = pd.read_csv('kather.csv')
    tokenized_dataset = make_token_dataset(text="Question: What's on the picture? Answer:",
                                             image_paths=df['image_path'].tolist(), 
                                             processor=processor, 
                                             tokenizer=tokenizer)

    activation_save_path = os.path.join(
        os.getenv('ACTIVATION_DATASET_DIR', 'activation_datasets'),
        args.model,
        args.entity_type
    )
    os.makedirs(activation_save_path, exist_ok=True)

    print(f'Begin processing {args.model} {args.entity_type}')

    layer_activations = get_layer_activations_hf(
        args, model, tokenized_dataset,
        device=args.device,
    )

    for layer_ix, activations in layer_activations.items():
        save_name = f'{args.entity_type}.{args.activation_aggregation}.{layer_ix}.pt'
        save_path = os.path.join(activation_save_path, save_name)
        activations = adjust_precision(
            activations.to(torch.float32), args.save_precision, per_channel=True)
        torch.save(activations, save_path)
