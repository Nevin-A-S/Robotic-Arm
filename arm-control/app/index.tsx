import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  SafeAreaView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
} from "react-native";
import Slider from "@react-native-community/slider";

// Type definitions
interface ServoPosition {
  index: number;
  angle: number;
}

interface WebSocketMessage {
  type:
    | "SET_SERVO"
    | "GET_POSITIONS"
    | "SAVE_POSITIONS"
    | "POSITIONS_UPDATE"
    | "SAVE_COMPLETE"
    | "RESET_POSITIONS";
  positions?: number[];
  index?: number;
  angle?: number;
  name?: string;
  filename?: string;
  success?: boolean;
}

const WEBSOCKET_URL = "ws://192.168.0.143:8000/ws";
const SERVO_COUNT = 6;
const MIN_ANGLE = 0;
const MAX_ANGLE = 180;

export default function App() {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [positions, setPositions] = useState<number[]>(
    Array(SERVO_COUNT).fill(90)
  );
  const [manualValues, setManualValues] = useState<string[]>(
    Array(SERVO_COUNT).fill("")
  );
  const [saveName, setSaveName] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const connectWebSocket = useCallback(() => {
    try {
      const websocket = new WebSocket(WEBSOCKET_URL);

      websocket.onopen = () => {
        setConnected(true);
        websocket.send(JSON.stringify({ type: "GET_POSITIONS" }));
      };

      websocket.onclose = () => {
        setConnected(false);
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      websocket.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);

          switch (data.type) {
            case "POSITIONS_UPDATE":
              if (data.positions) {
                setPositions(data.positions);
                setIsLoading(false);
              }
              if (data.success === false) {
                Alert.alert("Error", "Failed to update servo position");
              }
              break;
            case "SAVE_COMPLETE":
              if (data.filename) {
                Alert.alert("Success", `Positions saved to ${data.filename}`);
              } else {
                Alert.alert("Error", "Failed to save positions");
              }
              setIsLoading(false);
              break;
          }
        } catch (error) {
          console.error("WebSocket message error:", error);
          setIsLoading(false);
        }
      };

      ws.current = websocket;
    } catch (error) {
      console.error("WebSocket connection error:", error);
      setConnected(false);
    }
  }, []);

  const handleSliderChange = useCallback((index: number, value: number) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      setIsLoading(true);
      ws.current.send(
        JSON.stringify({
          type: "SET_SERVO",
          index,
          angle: Math.round(value),
        })
      );
    }
  }, []);

  const resetPositions = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      setIsLoading(true);
      ws.current.send(
        JSON.stringify({
          type: "RESET_POSITIONS",
        })
      );
    }
  }, []);

  const getPositions = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      setIsLoading(true);
      ws.current.send(
        JSON.stringify({
          type: "GET_POSITIONS",
        })
      );
    }
  }, []);

  const handleManualInput = useCallback(
    (index: number, value: string) => {
      const newValues = [...manualValues];
      newValues[index] = value.replace(/[^0-9]/g, "");
      setManualValues(newValues);
    },
    [manualValues]
  );

  const applyManualValue = useCallback(
    (index: number) => {
      const value = parseInt(manualValues[index]);
      if (!isNaN(value) && value >= MIN_ANGLE && value <= MAX_ANGLE) {
        handleSliderChange(index, value);
        setManualValues((prev) => {
          const newValues = [...prev];
          newValues[index] = "";
          return newValues;
        });
      } else {
        Alert.alert(
          "Invalid Angle",
          `Please enter a value between ${MIN_ANGLE} and ${MAX_ANGLE}`
        );
      }
    },
    [manualValues, handleSliderChange]
  );

  const savePositions = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      if (!saveName.trim()) {
        Alert.alert("Error", "Please enter a name for this save");
        return;
      }
      setIsLoading(true);
      ws.current.send(
        JSON.stringify({
          type: "SAVE_POSITIONS",
          name: saveName.trim(),
        })
      );
      setSaveName("");
    }
  }, [saveName]);

  const renderServoControl = useCallback(
    (index: number) => (
      <View key={index} style={styles.servoCard}>
        <View style={styles.servoHeader}>
          <Text style={styles.servoLabel}>Servo {index + 1}</Text>
          <Text style={styles.angleText}>{positions[index]}°</Text>
        </View>

        <Slider
          style={styles.slider}
          minimumValue={MIN_ANGLE}
          maximumValue={MAX_ANGLE}
          value={positions[index]}
          onSlidingComplete={(value) => handleSliderChange(index, value)}
          minimumTrackTintColor="#2196F3"
          maximumTrackTintColor="#DEDEDE"
          thumbTintColor="#2196F3"
        />

        <View style={styles.manualInput}>
          <TextInput
            style={styles.input}
            keyboardType="numeric"
            value={manualValues[index]}
            onChangeText={(value) => handleManualInput(index, value)}
            placeholder="Enter angle (0-180)"
            maxLength={3}
          />
          <TouchableOpacity
            style={styles.setButton}
            onPress={() => applyManualValue(index)}
          >
            <Text style={styles.buttonText}>Set</Text>
          </TouchableOpacity>
        </View>
      </View>
    ),
    [
      positions,
      manualValues,
      handleSliderChange,
      handleManualInput,
      applyManualValue,
    ]
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Servo</Text>
        <View style={styles.headerRight}>
          <TouchableOpacity
            style={styles.getPositionsButton}
            onPress={getPositions}
          >
            <Text style={styles.buttonText}>Get Positions</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.resetButton} onPress={resetPositions}>
            <Text style={styles.resetButtonText}>Reset All</Text>
          </TouchableOpacity>
          <View
            style={[
              styles.statusIndicator,
              { backgroundColor: connected ? "#4CAF50" : "#F44336" },
            ]}
          />
        </View>
      </View>

      {isLoading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#2196F3" />
        </View>
      )}

      <ScrollView style={styles.content}>
        {Array.from({ length: SERVO_COUNT }).map((_, index) =>
          renderServoControl(index)
        )}

        <View style={styles.saveSection}>
          <TextInput
            style={styles.saveInput}
            value={saveName}
            onChangeText={setSaveName}
            placeholder="Enter name for this position set"
          />
          <TouchableOpacity
            style={[
              styles.saveButton,
              !saveName.trim() && styles.saveButtonDisabled,
            ]}
            onPress={savePositions}
            disabled={!saveName.trim()}
          >
            <View style={styles.saveIcon}>
              <View style={styles.saveIconCircle} />
              <View style={styles.saveIconRect} />
            </View>
            <Text style={styles.saveButtonText}>Save Positions</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F5F5F5",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: 16,
    backgroundColor: "white",
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center",
  },
  getPositionsButton: {
    backgroundColor: "#2196F3",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    marginRight: 12,
  },
  resetButton: {
    backgroundColor: "#FF5722",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    marginRight: 12,
  },
  resetButtonText: {
    color: "white",
    fontSize: 14,
    fontWeight: "500",
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#333",
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  content: {
    padding: 16,
  },
  servoCard: {
    backgroundColor: "white",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  servoHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  servoLabel: {
    fontSize: 18,
    fontWeight: "600",
    color: "#333",
  },
  angleText: {
    fontSize: 16,
    color: "#2196F3",
    fontWeight: "500",
  },
  slider: {
    width: "100%",
    height: 40,
    marginBottom: 8,
  },
  manualInput: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 8,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 8,
    padding: 12,
    marginRight: 8,
    fontSize: 16,
    backgroundColor: "#F8F8F8",
  },
  setButton: {
    backgroundColor: "#2196F3",
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: "white",
    fontSize: 14,
    fontWeight: "500",
  },
  saveSection: {
    marginTop: 8,
    marginBottom: 32,
  },
  saveInput: {
    borderWidth: 1,
    borderColor: "#E0E0E0",
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    fontSize: 16,
    backgroundColor: "white",
  },
  saveButton: {
    backgroundColor: "#4CAF50",
    padding: 16,
    borderRadius: 8,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  saveButtonDisabled: {
    backgroundColor: "#BDBDBD",
  },
  saveButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "500",
    marginLeft: 8,
  },
  loadingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(255, 255, 255, 0.7)",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
  },
  saveIcon: {
    width: 20,
    height: 20,
    marginRight: 8,
  },
  saveIconCircle: {
    position: "absolute",
    top: 0,
    left: 0,
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: "white",
  },
  saveIconRect: {
    position: "absolute",
    bottom: 0,
    right: 0,
    width: 10,
    height: 10,
    backgroundColor: "white",
    borderRadius: 2,
  },
});
