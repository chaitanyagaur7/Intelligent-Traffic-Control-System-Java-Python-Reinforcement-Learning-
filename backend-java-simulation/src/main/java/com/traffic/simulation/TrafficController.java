package com.traffic.simulation;

import org.springframework.web.bind.annotation.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@RestController
@RequestMapping("/api/traffic")
@CrossOrigin(origins = "*")
public class TrafficController {

    // --- STATE: 4 Directions ---
    private int northCars = 5;
    private int southCars = 5;
    private int eastCars = 5;
    private int westCars = 5;
    
    // "VERTICAL" = North/South Green, "HORIZONTAL" = East/West Green
    private String currentAxis = "VERTICAL"; 

    @GetMapping("/state")
    public Map<String, Object> getState() {
        addRandomTraffic();
        Map<String, Object> state = new HashMap<>();
        state.put("north_cars", northCars);
        state.put("south_cars", southCars);
        state.put("east_cars", eastCars);
        state.put("west_cars", westCars);
        state.put("current_axis", currentAxis);
        return state;
    }

    @PostMapping("/action")
    public String performAction(@RequestBody Map<String, Integer> actionMap) {
        int action = actionMap.getOrDefault("action", 0);

        // Action 1 = Switch Axis (Vertical <-> Horizontal)
        if (action == 1) {
            if (currentAxis.equals("VERTICAL")) currentAxis = "HORIZONTAL";
            else currentAxis = "VERTICAL";
        }

        processTrafficFlow();
        return "Executed";
    }

    private void processTrafficFlow() {
        // If VERTICAL is Green, move North and South cars
        if (currentAxis.equals("VERTICAL")) {
            northCars = Math.max(0, northCars - 5);
            southCars = Math.max(0, southCars - 5);
        } 
        // If HORIZONTAL is Green, move East and West cars
        else {
            eastCars = Math.max(0, eastCars - 5);
            westCars = Math.max(0, westCars - 5);
        }
    }

    private void addRandomTraffic() {
        Random rand = new Random();
        northCars += rand.nextInt(3);
        southCars += rand.nextInt(3);
        eastCars += rand.nextInt(3);
        westCars += rand.nextInt(3);
    }
}