time python scripts/train_gemma_preprocess_dynamic.py \
  --jax-platform tpu \
  --max-dynamic-batch 128 \
  --sample-image-url "https://datasets-server.huggingface.co/cached-assets/unsloth/LaTeX_OCR/--/4da395c8a0253f4f30983cf08f2480f9bafbd561/--/default/test/0/image/image.jpg?Expires=1759280323&Signature=OvxxL6xoei6hH8vQS7PVHGFPwtO-NckPWnsLzBNpipmgig0Xx232JdKWk4YAVoSZB5awwatjCtvUrNSlbgjT5F2junqoPV6OhOLhXmmsPRFxEfaQId5zur5qi346r2yO99DtDUIvPjytmZnekQ2CMWdXzMDnYNZbeDXX88HhVss6VYneQvlerrA8uxtycrgeHz2LzpTx~yEEkXG6LQ~cCqykMThuyDmCqjSlCSUVix7NeNr8bEeQtdfPsk-u4FKqU1SMm~VtI1VRIaxxg54JoffeK-JeuDnXzhp7fz1Iro4PXwACyK06vGxgZvAfSs2EqgqamAqVIfrOxj9cJm11~Q__&Key-Pair-Id=K3EI6M078Z3AC3" \
  --sample-prompt "Convert the equation images to LaTeX equations. <start_of_image>" \
  --train-split "train[:3000]" \
  --train-epochs 5 \
  --eval-split "test[:3000]" \
  --eval-epochs 5
