import React, { useState } from 'react';
import Navbar from '../Navbar';
import Footer from '../Footer';
import BigFive from '../../Assets/BigFive3.jpg';
import '../../App.css'; // Import the CSS file

// Rest of your code...


const Personality = () => {
  const [question1, setQuestion1] = useState(3);
  const [question2, setQuestion2] = useState(3);
  const [question3, setQuestion3] = useState(3);
  const [question4, setQuestion4] = useState(3);
  const [question5, setQuestion5] = useState(3);

  return (
    <div className="home-container">
      <Navbar />
      <div className="home-banner-container">
        <div className="home-bannerImage-container">
          {/* <img src={BannerImage} alt="" /> */}
        </div>
        <form action='/predict' method="post">
          <div>
            <div className="contact-form-container">
              <p>I am the life of the party.</p>
            </div>
            <div className="contact-form-container">
              <input
                type="range"
                min="1"
                max="5"
                value={question1}
                onChange={(e) => setQuestion1(parseInt(e.target.value))}
              />
              <span>{question1}</span>
            </div>

            <div className="contact-form-container">
              <p>I don't talk a lot.</p>
            </div>
            <div className="contact-form-container">
              <input
                type="range"
                min="1"
                max="5"
                value={question2}
                onChange={(e) => setQuestion2(parseInt(e.target.value))}
              />
              <span>{question2}</span>
            </div>

            <div className="contact-form-container">
              <p>I feel comfortable around people.</p>
            </div>
            <div className="contact-form-container">
              <input
                type="range"
                min="1"
                max="5"
                value={question3}
                onChange={(e) => setQuestion3(parseInt(e.target.value))}
              />
              <span>{question3}</span>
            </div>

            <div className="contact-form-container">
              <p>I keep in the background.</p>
            </div>
            <div className="contact-form-container">
              <input
                type="range"
                min="1"
                max="5"
                value={question4}
                onChange={(e) => setQuestion4(parseInt(e.target.value))}
              />
              <span>{question4}</span>
            </div>

            <div className="contact-form-container">
              <p>I start conversations.</p>
            </div>
            <div className="contact-form-container">
              <input
                type="range"
                min="1"
                max="5"
                value={question5}
                onChange={(e) => setQuestion5(parseInt(e.target.value))}
              />
              <span>{question5}</span>
            </div>
          </div>
        </form>
      </div>
      <Footer />
    </div>
  );
};

export default Personality;
