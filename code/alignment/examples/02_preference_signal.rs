use rust_ml_alignment::{
    ChosenResponse, Instruction, PreferencePair, PreferenceRewards, RejectedResponse, Response,
    RewardScore, SignalSource,
};

fn main() -> Result<(), rust_ml_alignment::Error> {
    let pair = PreferencePair::new(
        Instruction::try_from("solve 2 + 2 with a visible check")?,
        ChosenResponse::from_response(Response::try_from("2 + 2 = 4")?),
        RejectedResponse::from_response(Response::try_from("2 + 2 = 5")?),
        SignalSource::try_from("public-preference-fixture")?,
    )?;
    let rewards =
        PreferenceRewards::new(RewardScore::try_from(0.80)?, RewardScore::try_from(-0.20)?);
    let signal = (pair + rewards)?;

    println!("chosen reward   = {}", signal.chosen_reward());
    println!("rejected reward = {}", signal.rejected_reward());
    println!("margin          = {}", signal.margin());

    Ok(())
}
