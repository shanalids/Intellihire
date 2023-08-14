import React from 'react'
import Navbar from '../Navbar'
import Footer from '../Footer'
import BigFive from "../../Assets/BigFive3.jpg";
import { FiArrowRight } from "react-icons/fi";

const Personality = () => {
    return (
        <div className="home-container">
            <Navbar />
            <div className="home-banner-container">
                <div className="home-bannerImage-container">
                    {/* <img src={BannerImage} alt="" /> */}
                </div>
                <div className="home-text-section">
                    <h1 className="primary-heading">
                        Big Five Trait Prediction
                    </h1>
                    <button className="secondary-button">
                        Get Started <FiArrowRight />{" "}
                    </button>
                </div>
                <div className="home-image-section">
                    <img src={BigFive} alt="" />
                </div>
            </div>
            {/* <Footer /> */}
        </div>
    )
}

export default Personality