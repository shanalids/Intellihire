import React, { useState } from 'react';
import * as XLSX from "xlsx";
import Navbar from '../Navbar';
import Footer from '../Footer';
import BigFive from '../../Assets/BigFive3.jpg';
import '../../App.css'; // Import the CSS file

const Personality = () => {
  const [data, setData] = useState([]);

  const handleFileUpload = (e) => {
    const reader = new FileReader();
    reader.readAsBinaryString(e.target.files[0]);
    reader.onload = (e) => {
      const data = e.target.result;
      const workbook = XLSX.read(data, { type: "binary" });
      const sheetName = workbook.SheetNames[0];
      const sheet = workbook.Sheets[sheetName];
      const parsedData = XLSX.utils.sheet_to_json(sheet);
      setData(parsedData);
    };
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission logic here
    // You can access the table data in the 'data' state variable
    console.log(data);
  };

  return (
    <div className="personalityMCQ">
      <Navbar />
      <br></br>

      <input
        type="file"
        accept=".xlsx, .xls"
        onChange={handleFileUpload}
      />
      <br></br>

      {data.length > 0 && (
        <form onSubmit={handleSubmit}>
          <table className="table">
            <thead>
              <tr>
                {Object.keys(data[0]).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr key={index}>
                  {Object.values(row).map((value, index) => (
                    <td key={index}>
                      <input type="text" value={value} name={`data${index}${Object.keys(row)[index]}`} readOnly />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <button type="submit">Submit</button>
        </form>
      )}
    </div>
  );
};

export default Personality;
