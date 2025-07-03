# Create BOS token embedding, falling back to EOS if BOS is not defined
start_token_id = decoder.tokenizer.bos_token_id if decoder.tokenizer.bos_token_id is not None else decoder.tokenizer.eos_token_id
bos_token_tensor = torch.tensor([[start_token_id]], device=image_input.device) # Ensure device matches
bos_embedding = decoder.qwen_model.get_input_embeddings()(bos_token_tensor)

# This is the starting point for the decoder
decoder_input_embeddings = torch.cat([image_input, bos_embedding], dim=1)

# Use beam search for higher quality generation 