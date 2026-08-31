# Prompt templates

`baseline_prompt_template.txt` is the V0 prompt: no retrieved context, and no list
of valid queues. Withholding the queue names is deliberate. Knowing which queues an
organisation operates is exactly the local knowledge retrieval is meant to supply,
so handing it to the baseline would hide the effect being measured.

`improved_rag_prompt_template.txt` is the V2 prompt, shown with a real retrieval
filled in. It adds the retrieved cases with their queue, type and priority labels,
a ranked candidate list, and the rule that the queue must be chosen from that list.

The ticket text is replaced by `{{TICKET_SUBJECT_AND_BODY}}` in both.
