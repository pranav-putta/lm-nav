import wandb
import os

def main():
    resume_run_id = input("Resume run id: ") 
    if resume_run_id:
        wandb.init(id=resume_run_id, resume='must',
                   entity='cvmlp-embodiment-transfer', project='lmnav')
    else:
        raise NotImplementedError('must have a run id rn')

    name = input("Artifact name: ")
    atype = input("Artifact type: ")
    filepath = input("Artifact filepath: ")

    path = os.path.abspath(filepath)

    artifact = wandb.Artifact(name, atype)
    artifact.add_reference(f'file://{path}', checksum=False, max_objects=1_000_000)
    wandb.log_artifact(artifact)

    wandb.finish()
    

if __name__ == "__main__":
    main()
