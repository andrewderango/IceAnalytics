import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface GameData {
  id: number;
  easternStartTime: string;
  gameNumber: number;
  gameStateId: number;
  period: number | null;
  homeScore: number;
  homeTeamId: number;
  visitingScore: number;
  visitingTeamId: number;
  gameType: number;
}

interface GameUpdate {
  game_id: number;
  home_score: number;
  visitor_score: number;
  home_prob: number;
  visitor_prob: number;
  overtime_prob: number;
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    // Initialize Supabase client with credentials from .env
    const supabaseUrl = Deno.env.get("REACT_APP_SUPABASE_PROJ_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || Deno.env.get("REACT_APP_SUPABASE_ANON_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Get the current season year (e.g., 2024 for 2024-2025 season)
    const now = new Date();
    const currentYear = now.getFullYear();
    const currentMonth = now.getMonth() + 1; // JavaScript months are 0-indexed
    
    // NHL season typically starts in October, so if we're before October, use previous year
    const seasonStartYear = currentMonth >= 10 ? currentYear : currentYear - 1;
    const seasonEndYear = seasonStartYear + 1;
    const seasonId = `${seasonStartYear}${seasonEndYear}`;

    console.log(`Fetching games for season ${seasonStartYear}-${seasonEndYear}`);

    // Fetch games from NHL API
    const gamesResponse = await fetch(
      `https://api.nhle.com/stats/rest/en/game?cayenneExp=season=${seasonId}`
    );

    if (!gamesResponse.ok) {
      throw new Error(`Failed to fetch games: ${gamesResponse.statusText}`);
    }

    const gamesData = await gamesResponse.json();
    const games: GameData[] = gamesData.data;

    console.log(`Fetched ${games.length} games`);

    // Get yesterday's date to filter games from previous day
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split("T")[0];

    console.log(`Looking for games from ${yesterdayStr}`);

    // Filter for regular season games from yesterday and calculate updates
    const gameUpdates: GameUpdate[] = games
      .filter((game) => {
        if (game.gameType !== 2) return false; // Only regular season
        
        const gameDate = new Date(game.easternStartTime);
        const gameDateStr = gameDate.toISOString().split("T")[0];
        
        return gameDateStr === yesterdayStr;
      })
      .filter((game) => game.gameStateId >= 6) // Only completed games (gameStateId 6 = Final, 7 = Final/OT)
      .map((game) => {
        // Determine winner and calculate probabilities
        let home_prob = 0.0;
        let visitor_prob = 0.0;
        let overtime_prob = 0.0;

        if (game.homeScore > game.visitingScore) {
          home_prob = 1.0;
          visitor_prob = 0.0;
        } else if (game.visitingScore > game.homeScore) {
          home_prob = 0.0;
          visitor_prob = 1.0;
        }

        // Check if game went to overtime/shootout
        // gameStateId 7 = Final/OT or Final/SO
        if (game.gameStateId === 7 || (game.period && game.period > 3)) {
          overtime_prob = 1.0;
        }

        return {
          game_id: game.id,
          home_score: game.homeScore,
          visitor_score: game.visitingScore,
          home_prob,
          visitor_prob,
          overtime_prob,
        };
      });

    console.log(
      `Found ${gameUpdates.length} completed games from yesterday (${yesterdayStr})`
    );

    if (gameUpdates.length === 0) {
      return new Response(
        JSON.stringify({
          success: true,
          message: `No completed games found for ${yesterdayStr}`,
          gamesProcessed: 0,
        }),
        {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
          status: 200,
        }
      );
    }

    // Update each game individually in the game_projections table
    let updatedCount = 0;
    const errors: string[] = [];

    for (const gameUpdate of gameUpdates) {
      const { error } = await supabase
        .from("game_projections")
        .update({
          home_score: gameUpdate.home_score,
          visitor_score: gameUpdate.visitor_score,
          home_prob: gameUpdate.home_prob,
          visitor_prob: gameUpdate.visitor_prob,
          overtime_prob: gameUpdate.overtime_prob,
        })
        .eq("game_id", gameUpdate.game_id);

      if (error) {
        console.error(`Error updating game ${gameUpdate.game_id}:`, error);
        errors.push(`Game ${gameUpdate.game_id}: ${error.message}`);
      } else {
        updatedCount++;
        console.log(
          `Updated game ${gameUpdate.game_id}: ${gameUpdate.visitor_score}-${gameUpdate.home_score}, Winner: ${gameUpdate.home_prob === 1.0 ? "Home" : "Visitor"}, OT: ${gameUpdate.overtime_prob === 1.0 ? "Yes" : "No"}`
        );
      }
    }

    if (errors.length > 0) {
      console.error(`Errors occurred: ${errors.join("; ")}`);
    }

    console.log(
      `Successfully updated ${updatedCount}/${gameUpdates.length} games in game_projections table`
    );

    return new Response(
      JSON.stringify({
        success: updatedCount > 0,
        message: `Successfully updated ${updatedCount}/${gameUpdates.length} games from ${yesterdayStr}`,
        gamesProcessed: updatedCount,
        gamesAttempted: gameUpdates.length,
        date: yesterdayStr,
        errors: errors.length > 0 ? errors : undefined,
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: updatedCount > 0 ? 200 : 207, // 207 = Multi-Status if some failed
      }
    );
  } catch (error) {
    console.error("Error in daily-games-scraper:", error);

    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 500,
      }
    );
  }
});
