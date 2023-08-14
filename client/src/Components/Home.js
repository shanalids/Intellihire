import React from "react";
import hrBackground from "../Assets/hr.jpeg";
// import BannerImage from "../Assets/bannerBg.jpg";
import Navbar from "./Navbar";
import Footer from "./Footer";
import { FiArrowRight } from "react-icons/fi";

const Home = () => {
    return (
        <div className="home-container">
            <Navbar />
            <div className="home-banner-container">
                <div className="home-bannerImage-container">
                    {/* <img src={BannerImage} alt="" /> */}
                </div>
                <div className="home-text-section">
                    <h1 className="primary-heading">
                        Recruit smarter, not harder.
                    </h1>
                    <p className="primary-text">
                        Simplify every step, from job postings to candidate screening revolutionizing your hiring process.
                    </p>
                    <button className="secondary-button">
                        Get Started <FiArrowRight />{" "}
                    </button>
                </div>
                <div className="home-image-section">
                    <img src={hrBackground} alt="" />
                </div>
            </div>
            {/* <Footer /> */}
        </div>
    )
}

export default Home