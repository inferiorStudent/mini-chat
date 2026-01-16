from trainer.pretrain import pretrain
from trainer.train_dpo import train_dpo
from trainer.train_full_sft import sft_train
from scripts.chat_terminal import chat, chat_, chat_origin
from trainer.train_ppo import train_ppo

if __name__ == "__main__":
    pretrain()
    sft_train()
    # train_dpo()
    # chat_()
    train_ppo()