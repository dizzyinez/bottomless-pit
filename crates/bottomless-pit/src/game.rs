use raylib::prelude::*;
use utils::{GameState, Point, Collide};
use input_handler::Inputs;
use super::player::player::Player;

#[derive(Debug)]
pub struct Game { 
    state: GameState,
    player: Option<Player>,
    keys: Inputs,
}

impl Game {
    pub fn new(keys: Inputs, rl: &mut RaylibHandle, thread: &RaylibThread) -> Self {
        let player = Player::make_baller(rl, thread, Point{x:10, y:100});
        Self {
            state: GameState::InGame,
            player: Some(player),
            keys,
        }
    }

    pub fn update(&mut self, rl: &mut RaylibHandle) {
        // the logic loop for the game
        match self.state {
            GameState::InGame => {
                self.ingame_update(rl);
            },
        }
    }

    pub fn draw(&self, mut d_handle: RaylibDrawHandle) {
        // the drawing loop for the game

        match &self.player {
            Some(player) => {
                player.draw(&mut d_handle);
                draw_healthbar(player, &mut d_handle);
            }
            None => {},
        }
    }

    // quick and dirty way to put stuff for testing
    fn ingame_update(&mut self, rl: &mut RaylibHandle) {
        self.player.as_mut()
            .unwrap()
            .update(rl, &self.keys);
    }

    // for now just feeding it an hitbox to check
    fn player_collision_check(&mut self) {
        match self.player.as_mut() {
            Some(p) => {
                let attack = p.get_active_attack();
                let hurtbox = p.get_hurtbox();
                let hit = Collide::ploy_poly(hurtbox.get_poly(), &Vec::new());
            },
            None => {},
        }
    }
}

fn draw_healthbar(player: &Player, d_handle: &mut RaylibDrawHandle) {
    // assume 100 = 100% fill should take up 1/2 or 1/3 of the screen?
    let window_width = d_handle.get_screen_width();
    let hp = player.get_health();
    let fill = ((window_width / 3)as f32 * (hp/100.0)).round() as i32;

    d_handle.draw_rectangle(20, 20, fill, 40, Color::RED);
    d_handle.draw_rectangle_lines(20, 20, window_width / 3, 40, Color::BLACK);
}