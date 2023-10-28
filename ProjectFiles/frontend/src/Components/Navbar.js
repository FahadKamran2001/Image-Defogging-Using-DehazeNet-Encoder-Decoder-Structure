import { NavLink, useLocation } from "react-router-dom";
import Home from "./Home";
import "./Navbar.css"
import homeLogo from "../media/Logo.png"

function Navbar()
{
    const location = useLocation();
    console.log(location.pathname);
    return(
        <>
            <div className="navbar">
                <div className="navbarLogo">
                    <img src={homeLogo} style={{width:"15rem"}} alt="Logo"/>
                </div>
                <div className="navbarLinks">
                    <ul>
                        <li className={location.pathname == "/" ? "activeNav" : ""}>
                            <NavLink to="/">
                                Home
                            </NavLink>
                        </li>
                        <li className={location.pathname == "/Research" ? "activeNav" : ""}>
                            <NavLink to="/Research">
                                Research
                            </NavLink>
                        </li>
                        <li className={location.pathname == "/Team" ? "activeNav" : ""}>
                            <NavLink to="/Team">
                                Team
                            </NavLink>
                        </li>
                        <li className={location.pathname == "/About" ? "activeNav" : ""}>
                            <NavLink to="/About">
                                About
                            </NavLink>
                        </li>
                        <a className="contactUsBtn" href="https://www.linkedin.com/company/aim-lab/mycompany/" target="_blank">
                            <li >
                                Contact Us
                            </li>
                        </a>
                        
                    </ul>
                </div>
            </div>
        </>
    )
}


export default Navbar;