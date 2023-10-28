import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import Select, { ValueType } from "react-select";
import Loginpage from "./Components/Loginpage";
import {Routes, BrowserRouter,Route} from 'react-router-dom'
import Login from 'login.tsx'


const options: { value: string; label: string }[] = [
  { value: "object1", label: "Toys" },
  { value: "object2", label: "Accesories" },
  { value: "object3", label: "Clothes" },
  { value: "object4", label: "Food" },
  { value: "object5", label: "Beverage" },
  { value: "object6", label: "Jwelley" },
  { value: "object7", label: "Kitchen-appliance" },
  // Add more objects as needed
];
z/

function App() {
  //login part
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");

  const handleLogin = (enteredUsername: string, password: string) => {
    // In a real application, you would perform authentication with a server.
    // For this example, we'll assume a hardcoded username and password.
    const validUsername = "yourUsername";
    const validPassword = "yourPassword";

    if (enteredUsername === validUsername && password === validPassword) {
      setIsLoggedIn(true);
      setUsername(enteredUsername);
    } else {
      alert("Invalid username or password");
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername("");
  };

  const [selectedObject, setSelectedObject] =
    useState<ValueType<{ value: string; label: string }>>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [boundingBox, setBoundingBox] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    const imageURL = URL.createObjectURL(file);
    setImageURL(imageURL);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  const handleObjectSelect = (
    selectedOption: ValueType<{ value: string; label: string }>
  ) => {
    setSelectedObject(selectedOption);
    // Perform object recognition here using a backend service or TensorFlow.js
    // Set the bounding box coordinates in the state.
    // You can use setBoundingBox for this.
  };

  return (
    <div className="App">
      {isLoggedIn ? (
        <div>
          <h1>Welcome, {username}!</h1>
          <button onClick={handleLogout}>Logout</button>
        </div>
      ) : (
        <Loginpage onLogin={handleLogin} />
      )}

      <h1>Object Recognition with Bounding Box</h1>
      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        <p>Drag & drop an image here, or click to select one</p>
      </div>
      {imageURL && <img src={imageURL} alt="Uploaded" />}
      <Select
        options={options}
        value={selectedObject}
        onChange={handleObjectSelect}
        placeholder="Select Object"
      />
      {boundingBox && (
        <div className="bounding-box">
          {/* Display the bounding box on the image */}
          {/* You can use CSS to create the bounding box */}
        </div>
      )}
    </div>
  );

  return (
    <div className="App">
      {isLoggedIn ? (
        <div>
          <h1>Welcome, {username}!</h1>
          <button onClick={handleLogout}>Logout</button>
        </div>
      ) : (
        <Loginpage onLogin={handleLogin} />
      )}
    </div>
  );
}

export default App;
