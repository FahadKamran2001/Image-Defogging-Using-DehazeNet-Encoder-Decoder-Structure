import "./App.css"
import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Home from "./Components/Home";
import Navbar from "./Components/Navbar";
import Footbar from "./Components/Footbar";
import Research from "./Components/Research";
import Team from "./Components/Team";
import About from "./Components/About";

function App() {

 

  return (
    <>
      
      <Router>
        <Navbar/>
        <Routes>
          <Route path="/" Component={Home}></Route>
          <Route path="/Research" Component={Research}></Route>
          <Route path="/Team" Component={Team}></Route>
          <Route path="/About" Component={About}></Route>
        </Routes>
        <Footbar/>
      </Router>
      
    </>
  );
}

export default App;
